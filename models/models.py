import torch
from torch import nn


class LorentzInvariantFeatures(nn.Module):
    def __init__(self, feature_set=['pt', 'eta', 'phi', 'm']):
        super().__init__()
        self.feature_set = feature_set
    
    def forward(self, four_momenta):
        # four_momenta shape: [..., 4] where last dim is (px, py, pz, E)
        # Calculate Lorentz invariant quantities
        mass_squared = four_momenta[..., 3]**2 - (four_momenta[..., 0]**2 + 
                                                 four_momenta[..., 1]**2 + 
                                                 four_momenta[..., 2]**2)
        pt = torch.sqrt(four_momenta[..., 0]**2 + four_momenta[..., 1]**2)
        eta = torch.asinh(four_momenta[..., 2] / pt)
        eta = torch.nan_to_num(eta, nan=0.0)
        phi = torch.atan2(four_momenta[..., 1], four_momenta[..., 0])
        
        features = []
        if 'm' in self.feature_set:
            features.append(mass_squared)
        if 'pt' in self.feature_set:
            features.append(pt)
        if 'eta' in self.feature_set:
            features.append(eta)
        if 'phi' in self.feature_set:
            features.append(phi)
        return torch.stack(features, dim=-1)


class TestNetwork(nn.Module):
    '''
    Create a network which first embeds our physics 4-vectors (& tag information, where relevant), plus a type 
    embedding (depending on if it's electron, muon, neutrino, sjet, ljet) into a residual stream of dimension
    hidden_dim.
    The embedding might be done directly on the px, py, pz, E, or we might transform to a Lorentz invariant Pt,
    eta, phi, m^2, and embed that.

    Then apply a series of attention blocks, where each block has:
     - a self-attention mechanism, with the option to project down either before the attention mechanism, 
        or bottleneck after attention, or neither. It also includes padding mask for objects which are not
        present in the event (these have type=num_particle_types-1)
        Note, if we use the bottleneck after attention, this is not handled in the forward function, but 
        rather must be handled by a special hook function which the user has to register (kinda hacky but 
        it works for now)
    - optionally a MLP after the attention mechanism
    - a residual skip connection around each of these components

    Finally, project the residual stream onto classes per-object, to predict presence in different intermediate
    states in our event (eg. is this object a decay product of a W boson?)
    '''
    def __init__(self, use_lorentz_invariant_features=True, bottleneck_attention=None, feature_set=['pt', 'eta', 'phi', 'm', 'tag'], num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None, hidden_dim_attn=None, num_particle_types=5, num_object_net_layers=1, is_layer_norm=False, is_reconstruction_model=True):
        super().__init__()
        self.bottleneck_attention = bottleneck_attention
        self.num_attention_blocks = num_attention_blocks
        self.include_mlp = include_mlp
        self.use_lorentz_invariant_features = use_lorentz_invariant_features
        self.num_particle_types = num_particle_types
        if hidden_dim_mlp is None:
            hidden_dim_mlp = hidden_dim
        self.hidden_dim_attn = hidden_dim_attn
        self.is_reconstruction_model = is_reconstruction_model

        if self.use_lorentz_invariant_features:
            self.invariant_features = LorentzInvariantFeatures(feature_set=feature_set)
        
        # Object type embedding
        self.type_embedding = nn.Embedding(self.num_particle_types, embedding_size)  # 5 object types
        
        # Initial per-object processing
        self.object_net = nn.Sequential(*[
            nn.Linear(len(feature_set) + embedding_size, hidden_dim),  # All features except type + type embedding
            # Now do one LayerNorm followed by one ReLU followed by one Linear for each object net layer. Has to be dynamic depending on num_object_net_layers
            *([nn.LayerNorm(hidden_dim)]*is_layer_norm+[nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)])*num_object_net_layers
        ]
        )
        
        # Create multiple attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.ModuleDict({
                **({f'input_projection': nn.Linear(hidden_dim, self.hidden_dim_attn)} if (self.hidden_dim_attn is not None) else {}), # Method one to optionally project to a different dimension for self-attention
                'self_attention': nn.MultiheadAttention(
                    embed_dim=self.hidden_dim_attn if (self.hidden_dim_attn is not None) else hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                ),
                **({f'bottleneck_down': nn.ModuleList([
                    nn.Linear(hidden_dim, self.bottleneck_attention)
                    for _ in range(num_heads)])} if (self.bottleneck_attention is not None) else {}), # Method two to optionally project to a different dimension for bottleneck attention
                **({f'bottleneck_up': nn.ModuleList([
                    nn.Linear(self.bottleneck_attention, hidden_dim)
                    for _ in range(num_heads)])} if (self.bottleneck_attention is not None) else {}), # Optionally project back to the original dimension (if we use method 2)
                **({f'output_projection': nn.Linear(self.hidden_dim_attn, hidden_dim)} if (self.hidden_dim_attn is not None) else {}), # Optionally project back to the original model dimension (if we use method 1)
                **({'post_attention': nn.Sequential( # Optional MLP later after attention
                    nn.Linear(hidden_dim, hidden_dim_mlp),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim_mlp, hidden_dim),
                )} if self.include_mlp else {})
            }) for _ in range(num_attention_blocks)
        ])
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features, object_types):
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if self.use_lorentz_invariant_features:
            invariant_features = self.invariant_features(object_features[...,:4])
        else:
            invariant_features = object_features[...,:4]
        combined = torch.cat([invariant_features, object_features[...,4:], type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Apply attention blocks
        for block in self.attention_blocks:
            # Store original features for residual connection
            identity = object_features
            # Apply self-attention
            if self.hidden_dim_attn is not None: # Project input to attention dimension
                projected_input = block['input_projection'](object_features)
            else:
                projected_input = object_features
            attention_output, _ = block['self_attention'](
                projected_input, projected_input, projected_input,
                key_padding_mask=(object_types==(self.num_particle_types-1))
            )
            if self.hidden_dim_attn is not None:
                # Project output back to original dimension
                attention_output = block['output_projection'](attention_output)
            # Add residual connection
            if self.include_mlp:
                residual = identity + attention_output
                identity = residual
                # Post-attention processing
                mlp_output = block['post_attention'](residual)
                object_features = identity + mlp_output
            else:
                object_features = identity + attention_output
        if not self.is_reconstruction_model:
            # This is a classificaiton model, so we need to pool the object features to get a single vector per event (ie the event background, signal1, signal2, etc?)
            # Pool by taking mean of non-padding to ensure permutation invariance and invariance to number of objects.
            pooled = torch.sum(object_features, dim=1) / torch.sum(object_types!=(self.num_particle_types-1), dim=-1).unsqueeze(-1)
            return self.classifier(pooled)
        else:
            # This is a reconstruction model, so we need to return the class per object (ie, which intermediate state is this object a decay product of?)
            return self.classifier(object_features)
