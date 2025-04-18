import torch
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn

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


class DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(nn.Module):
        def __init__(self, use_lorentz_invariant_features=True, bottleneck_attention=None, feature_set=['pt', 'eta', 'phi', 'm', 'tag'], num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None, num_particle_types=5):
            super().__init__()
            self.bottleneck_attention = bottleneck_attention
            self.num_attention_blocks = num_attention_blocks
            self.include_mlp = include_mlp
            self.use_lorentz_invariant_features = use_lorentz_invariant_features
            self.num_particle_types = num_particle_types
            if hidden_dim_mlp is None:
                hidden_dim_mlp = hidden_dim

            if self.use_lorentz_invariant_features:
                self.invariant_features = LorentzInvariantFeatures(feature_set=feature_set)
            
            # Object type embedding
            self.type_embedding = nn.Embedding(self.num_particle_types, embedding_size)  # 5 object types
            
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(len(feature_set) + embedding_size, hidden_dim),  # All features except type + type embedding
                # nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Create multiple attention blocks
            self.attention_blocks = nn.ModuleList([
                nn.ModuleDict({
                    # 'layer_norm1': nn.LayerNorm(hidden_dim),
                    'self_attention': nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=0.0,
                        batch_first=True,
                    ),
                    **({f'bottleneck_down': nn.ModuleList([
                        nn.Linear(hidden_dim, self.bottleneck_attention)
                        for _ in range(num_heads)])} if (self.bottleneck_attention is not None) else {}),
                    **({f'bottleneck_up': nn.ModuleList([
                        nn.Linear(self.bottleneck_attention, hidden_dim)
                        for _ in range(num_heads)])} if (self.bottleneck_attention is not None) else {}),
                    # 'layer_norm2': nn.LayerNorm(hidden_dim),
                    **({'post_attention': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim_mlp),
                        nn.GELU(),
                        nn.Dropout(dropout_p),
                        nn.Linear(hidden_dim_mlp, hidden_dim),
                    )} if self.include_mlp else {})
                }) for _ in range(num_attention_blocks)
            ])
            # Final classification layers
            self.classifier = nn.Sequential(
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                # nn.Dropout(dropout_p),
                # nn.Linear(hidden_dim, hidden_dim // 2),
                # nn.ReLU(),
                # nn.Linear(hidden_dim // 2, num_classes)
                nn.Linear(hidden_dim, num_classes)
            )
    
        def forward(self, object_features, object_types):
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(object_types)
            if self.use_lorentz_invariant_features:
                # invariant_features = self.invariant_features(object_features[...,:4])
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
                # normed_features = block['layer_norm1'](object_features)
                attention_output, _ = block['self_attention'](
                    object_features, object_features, object_features,
                    key_padding_mask=(object_types==(self.num_particle_types-1))
                )
                # Add residual connection and normalize
                if self.include_mlp:
                    residual = identity + attention_output
                    identity = residual
                    # normed_mlpin = block['layer_norm2'](residual)
                    # Post-attention processing
                    mlp_output = block['post_attention'](residual)
                    object_features = identity + mlp_output
                else:
                    object_features = identity + attention_output
            return self.classifier(object_features)





class DeepSetsWithResidualSelfAttentionVariableTrueSkipClassifier(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None):
            super().__init__()
            self.num_attention_blocks = num_attention_blocks
            self.include_mlp = include_mlp
            if hidden_dim_mlp is None:
                hidden_dim_mlp = hidden_dim

            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Create multiple attention blocks
            self.attention_blocks = nn.ModuleList([
                nn.ModuleDict({
                    # 'layer_norm1': nn.LayerNorm(hidden_dim),
                    'self_attention': nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout_p/2,
                        batch_first=True,
                    ),
                    'layer_norm2': nn.LayerNorm(hidden_dim),
                    **({'post_attention': nn.Sequential(
                        # nn.Linear(hidden_dim, hidden_dim_mlp),
                        # nn.GELU(),
                        # nn.Dropout(dropout_p),
                        # nn.Linear(hidden_dim_mlp, hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_p),
                    )} if self.include_mlp else {})
                }) for _ in range(num_attention_blocks)
            ])
            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
                # nn.Linear(hidden_dim, num_classes)
            )
    
        def forward(self, object_features, object_types):
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(object_types)
            if USE_LORENTZ_INVARIANT_FEATURES:
                invariant_features = self.invariant_features(object_features[...,:4])
            combined = torch.cat([invariant_features, object_features[...,4:], type_emb], dim=-1)
            # Process each object
            object_features = self.object_net(combined)
            # Apply attention blocks
            for block in self.attention_blocks:
                # Store original features for residual connection
                identity = object_features
                # Apply self-attention
                # normed_features = block['layer_norm1'](object_features)
                attention_output, _ = block['self_attention'](
                    object_features, object_features, object_features,
                    key_padding_mask=(object_types==(N_CTX-1))
                )
                # Add residual connection and normalize
                if self.include_mlp:
                    residual = identity + attention_output
                    # identity = residual
                    normed_mlpin = block['layer_norm2'](residual)
                    # Post-attention processing
                    mlp_output = block['post_attention'](normed_mlpin)
                    # object_features = identity + mlp_output
                    object_features = mlp_output
                else:
                    object_features = identity + attention_output

            # Pool by taking mean of non-padding to ensure permutation invariance
            pooled = torch.sum(object_features, dim=1) / torch.sum(object_types!=(N_CTX-1), dim=-1).unsqueeze(-1)
            return self.classifier(pooled)







###################################################
###############    Older models    ################
###################################################


USE_LORENTZ_INVARIANT_FEATURES=True
N_CTX=5
TAG_INFO_INPUT=True


class DeepSetsBasic(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3):
        super().__init__()
        self.num_attention_blocks = num_attention_blocks

        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Create multiple attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.ModuleDict({
                # 'self_attention': nn.MultiheadAttention(
                #     embed_dim=hidden_dim,
                #     num_heads=num_heads,
                #     dropout=0.0,
                #     batch_first=True,
                # ),
                'layer_norm': nn.LayerNorm(hidden_dim),
                'post_attention': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                )
            }) for _ in range(num_attention_blocks)
        ])
        # Final classification layers
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim // 2, num_classes)
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features, object_types):
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            invariant_features = self.invariant_features(object_features[...,:4])
        combined = torch.cat([invariant_features, object_features[...,4:], type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Apply attention blocks
        for block in self.attention_blocks:
            # Store original features for residual connection
            # identity = object_features
            # # Apply self-attention
            # attention_output, _ = block['self_attention'](
            #     object_features, object_features, object_features,
            #     key_padding_mask=(object_types==(N_CTX-1))
            # )
            # # Add residual connection and normalize
            # attention_output = identity + attention_output
            attention_output = block['layer_norm'](object_features)
            # Post-attention processing
            object_features = block['post_attention'](attention_output)
        return self.classifier(object_features)
class TfLensDeepsetsResidualSelfAttention(HookedTransformer):
    def __init__(self, is_categorical, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, embedding_size=32, num_layers=3, device='cpu', **kwargs):
        cfg = HookedTransformerConfig(
            d_model=hidden_dim,
            n_layers=num_layers,
            d_head= hidden_dim // num_heads,
            n_heads=num_heads,
            normalization_type='LN',
            n_ctx=N_CTX, # Max number of types of object per event + 1 because we want a dummy row in the embedding matrix for non-existing particles
            d_vocab=4+int(TAG_INFO_INPUT), # Number of inputs per object
            d_vocab_out=1,  # 1 because we're doing binary classification
            d_mlp=hidden_dim,
            attention_dir="bidirectional",  # defaults to "causal"
            act_fn="relu",
            use_attn_result=True,
            device=str(device),
            use_hook_tokens=True,
        )
        super(TfLensDeepsetsResidualSelfAttention, self).__init__(cfg, **kwargs)
        
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types

        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Final classification layers
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim // 2, num_classes)
            nn.Linear(hidden_dim, num_classes)
        )
        for hpn in ['hook_ObjectInputs', 'hook_LorentzInvariantObjectInputs', 'hook_ObjectTypes', 'hook_TransformerIns', 'hook_TransformerOuts']:
            self.hook_dict[hpn] = HookPoint()
            self.hook_dict[hpn].name = hpn
            self.mod_dict[hpn] = self.hook_dict[hpn]
        self.is_categorical = is_categorical
        if self.is_categorical:
            self.num_classes = num_classes
        
    def forward(self, tokens: Float[Tensor, "batch object d_input"], token_types: Float[Tensor, "batch object"], **kwargs) -> Float[Tensor, "batch d_model"]:
        # Do the embedding of object types
        self.hook_dict['hook_ObjectTypes'](token_types) # shape ["batch object"]
        type_emb = self.type_embedding(token_types) #  # shape ["batch num_embedding"]

        # Convert px, py, pz, E into MassSq, Pt, Eta, Phi
        self.hook_dict['hook_ObjectInputs'](tokens)
        if USE_LORENTZ_INVARIANT_FEATURES:
            invariant_features = self.invariant_features(tokens[...,:4])
        self.hook_dict['hook_LorentzInvariantObjectInputs'](invariant_features)
        
        # Encode these features/token-embedding into model-dimension
        combined = torch.cat([invariant_features, tokens[...,4:], type_emb], dim=-1)
        object_features = self.object_net(combined)
        self.hook_dict['hook_TransformerIns'](invariant_features)

        # Run the transformer bit of this model (skipping the embedding with start_at_layer and the unambedding with stop_at_layer)
        if ('start_at_layer'in kwargs) or ('stop_at_layer' in kwargs):
            raise NotImplementedError
        else:
            if self.is_categorical:
                transformer_outs = super(TfLensDeepsetsResidualSelfAttention, self).forward(object_features, start_at_layer=0, stop_at_layer=-1, **kwargs)
                self.hook_dict['hook_TransformerOuts'](transformer_outs)
                return self.classifier(transformer_outs)
            else:
                raise NotImplementedError
                # class_outs = super(MyHookedTransformer, self).forward(output, start_at_layer=0, stop_at_layer=-1, **kwargs)
                # return class_outs
class DeepSetsWithResidualSelfAttentionVariable(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3):
        super().__init__()
        self.num_attention_blocks = num_attention_blocks

        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Create multiple attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                ),
                # 'layer_norm': nn.LayerNorm(hidden_dim),
                'post_attention': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                )
            }) for _ in range(num_attention_blocks)
        ])
        # Final classification layers
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim // 2, num_classes)
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features, object_types):
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            invariant_features = self.invariant_features(object_features[...,:4])
        combined = torch.cat([invariant_features, object_features[...,4:], type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Apply attention blocks
        for block in self.attention_blocks:
            # Store original features for residual connection
            identity = object_features
            # Apply self-attention
            attention_output, _ = block['self_attention'](
                object_features, object_features, object_features,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Add residual connection and normalize
            attention_output = identity + attention_output
            # attention_output = block['layer_norm'](attention_output)
            # Post-attention processing
            object_features = block['post_attention'](attention_output)
        return self.classifier(object_features)
class DeepSetsWithResidualSelfAttentionVariableTrueSkip(nn.Module):
    def __init__(self, feature_set=['pt', 'eta', 'phi', 'm', 'tag'], num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None):
        super().__init__()
        self.num_attention_blocks = num_attention_blocks
        self.include_mlp = include_mlp
        if hidden_dim_mlp is None:
            hidden_dim_mlp = hidden_dim

        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures(feature_set=feature_set)
        
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(len(feature_set) + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Create multiple attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.ModuleDict({
                # 'layer_norm1': nn.LayerNorm(hidden_dim),
                'self_attention': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                ),
                # 'layer_norm2': nn.LayerNorm(hidden_dim),
                **({'post_attention': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim_mlp),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim_mlp, hidden_dim),
                )} if self.include_mlp else {})
            }) for _ in range(num_attention_blocks)
        ])
        # Final classification layers
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim // 2, num_classes)
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features, object_types):
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            # invariant_features = self.invariant_features(object_features[...,:4])
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
            # normed_features = block['layer_norm1'](object_features)
            attention_output, _ = block['self_attention'](
                object_features, object_features, object_features,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Add residual connection and normalize
            if self.include_mlp:
                residual = identity + attention_output
                identity = residual
                # normed_mlpin = block['layer_norm2'](residual)
                # Post-attention processing
                mlp_output = block['post_attention'](residual)
                object_features = identity + mlp_output
            else:
                object_features = identity + attention_output
        return self.classifier(object_features)
class GraphNN(nn.Module):
    assert(False)
    def __init__(self, feature_set=['pt', 'eta', 'phi', 'm', 'tag'], num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None):
        super().__init__()
        self.num_attention_blocks = num_attention_blocks
        self.include_mlp = include_mlp
        if hidden_dim_mlp is None:
            hidden_dim_mlp = hidden_dim

        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures(feature_set=feature_set)
        
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(len(feature_set) + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Create multiple attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.ModuleDict({
                # 'layer_norm1': nn.LayerNorm(hidden_dim),
                'self_attention': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                ),
                # 'layer_norm2': nn.LayerNorm(hidden_dim),
                **({'post_attention': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim_mlp),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim_mlp, hidden_dim),
                )} if self.include_mlp else {})
            }) for _ in range(num_attention_blocks)
        ])
        # Final classification layers
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim // 2, num_classes)
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features, object_types):
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            # invariant_features = self.invariant_features(object_features[...,:4])
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
            # normed_features = block['layer_norm1'](object_features)
            attention_output, _ = block['self_attention'](
                object_features, object_features, object_features,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Add residual connection and normalize
            if self.include_mlp:
                residual = identity + attention_output
                identity = residual
                # normed_mlpin = block['layer_norm2'](residual)
                # Post-attention processing
                mlp_output = block['post_attention'](residual)
                object_features = identity + mlp_output
            else:
                object_features = identity + attention_output
        return self.classifier(object_features)
class SingleSelfAttentionNet(nn.Module):
    def __init__(self, layer_norm_input=True, skip_connection=True, include_mlp=False, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=10, hidden_dim_mlp=None):
        super().__init__()
        self.include_mlp = include_mlp
        self.skip_connection = skip_connection
        if hidden_dim_mlp is None:
            hidden_dim_mlp = hidden_dim

        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        
        # Initial per-object processing
        if layer_norm_input:
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Create multiple attention blocks
        self.attention_block = nn.ModuleDict({
                # 'layer_norm1': nn.LayerNorm(hidden_dim),
                'self_attention': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                ),
                # 'layer_norm2': nn.LayerNorm(hidden_dim),
                **({'post_attention': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim_mlp),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim_mlp, hidden_dim),
                )} if self.include_mlp else {})
            }) 
        # Final classification layers
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout_p),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim // 2, num_classes)
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features, object_types):
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            invariant_features = self.invariant_features(object_features[...,:4])
        else:
            invariant_features = object_features[...,:4]
        combined = torch.cat([invariant_features, object_features[...,4:], type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Apply attention block
        # Store original features for residual connection
        identity = object_features
        # Apply self-attention
        # normed_features = block['layer_norm1'](object_features)
        attention_output, _ = self.attention_block['self_attention'](
            object_features, object_features, object_features,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Add residual connection and normalize
        if self.include_mlp:
            if self.skip_connection:
                residual = identity + attention_output
            else:
                residual = attention_output
            identity = residual
            # normed_mlpin = block['layer_norm2'](residual)
            # Post-attention processing
            mlp_output = self.attention_block['post_attention'](residual)
            if self.skip_connection:
                object_features = identity + mlp_output
            else:
                object_features = mlp_output
        else:
            if self.skip_connection:
                object_features = identity + attention_output
            else:
                object_features = attention_output
        return self.classifier(object_features)
class DeepSetsWithResidualSelfAttentionVariableLongclassifier(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, num_classifier_layers=1):
        super().__init__()
        self.num_attention_blocks = num_attention_blocks

        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Create multiple attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                ),
                'layer_norm': nn.LayerNorm(hidden_dim),
                'post_attention': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                )
            }) for _ in range(num_attention_blocks)
        ])
        # Final classification layers
        assert(num_classifier_layers>0)
        self.classifier = nn.Sequential(
            *[
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ] + [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ]*(num_classifier_layers-1) + [
                nn.Linear(hidden_dim, num_classes)
            ]
        )

    def forward(self, object_features, object_types):
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            invariant_features = self.invariant_features(object_features[...,:4])
        combined = torch.cat([invariant_features, object_features[...,4:], type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Apply attention blocks
        for block in self.attention_blocks:
            # Store original features for residual connection
            identity = object_features
            # Apply self-attention
            attention_output, _ = block['self_attention'](
                object_features, object_features, object_features,
                key_padding_mask=(object_types==(N_CTX-1))
            )
            # Add residual connection and normalize
            attention_output = identity + attention_output
            attention_output = block['layer_norm'](attention_output)
            # Post-attention processing
            object_features = block['post_attention'](attention_output)
        return self.classifier(object_features)

class DeepSetsWithResidualSelfAttentionTriple(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32):
        super().__init__()
        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Self-attention layer for object interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # dropout=dropout_p/2,
            dropout=0.0,
            batch_first=True,
        )
        self.self_attention2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # dropout=dropout_p/2,
            dropout=0.0,
            batch_first=True,
        )
        self.self_attention3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # dropout=dropout_p/2,
            dropout=0.0,
            batch_first=True,
        )
        # Processing after attention with normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        self.post_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.post_attention2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.post_attention3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, object_features, object_types):
        batch_size, num_objects, feature_dim = x.shape
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            # object_features[...,:4] = self.invariant_features(object_features[...,:4])
            invariant_features = self.invariant_features(object_features[...,:4])
        combined = torch.cat([invariant_features, object_features[...,4:], type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Store original features for residual connection
        identity = object_features
        # Apply self-attention to model interactions between objects
        # This creates a mechanism for objects to attend to each other
        attention_output, _ = self.self_attention(
            object_features, object_features, object_features,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Add residual connection and normalize
        attention_output = identity + attention_output
        attention_output = self.layer_norm(attention_output)
        # Post-attention processing
        attention_output = self.post_attention(attention_output)
        # Store original features for residual connection
        identity2 = attention_output
        # Apply self-attention to model interactions between objects
        # This creates a mechanism for objects to attend to each other
        attention_output2, _ = self.self_attention2(
            attention_output, attention_output, attention_output,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Add residual connection and normalize
        attention_output2 = identity2 + attention_output2
        attention_output2 = self.layer_norm2(attention_output2)
        # Post-attention processing
        attention_output2 = self.post_attention2(attention_output2)
        # Store original features for residual connection
        identity3 = attention_output2
        # Apply self-attention to model interactions between objects
        # This creates a mechanism for objects to attend to each other
        attention_output3, _ = self.self_attention3(
            attention_output2, attention_output2, attention_output2,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Add residual connection and normalize
        attention_output3 = identity3 + attention_output3
        attention_output3 = self.layer_norm3(attention_output3)
        # Post-attention processing
        attention_output3 = self.post_attention3(attention_output3)
        return self.classifier(attention_output3)

class DeepSetsWithResidualSelfAttentionDouble(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32):
        super().__init__()
        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Self-attention layer for object interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # dropout=dropout_p/2,
            dropout=0.0,
            batch_first=True,
        )
        self.self_attention2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # dropout=dropout_p/2,
            dropout=0.0,
            batch_first=True,
        )
        # Processing after attention with normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.post_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.post_attention2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, object_features, object_types):
        batch_size, num_objects, feature_dim = x.shape
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            object_features[...,:4] = self.invariant_features(object_features[...,:4])
        combined = torch.cat([object_features, type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Store original features for residual connection
        identity = object_features
        # Apply self-attention to model interactions between objects
        # This creates a mechanism for objects to attend to each other
        attention_output, _ = self.self_attention(
            object_features, object_features, object_features,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Add residual connection and normalize
        attention_output = identity + attention_output
        attention_output = self.layer_norm(attention_output)
        # Post-attention processing
        attention_output = self.post_attention(attention_output)
        # Store original features for residual connection
        identity2 = attention_output
        # Apply self-attention to model interactions between objects
        # This creates a mechanism for objects to attend to each other
        attention_output2, _ = self.self_attention2(
            attention_output, attention_output, attention_output,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Add residual connection and normalize
        attention_output2 = identity2 + attention_output2
        attention_output2 = self.layer_norm2(attention_output2)
        # Post-attention processing
        attention_output2 = self.post_attention2(attention_output2)
        return self.classifier(attention_output2)

class DeepSetsWithResidualSelfAttention(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32):
        super().__init__()
        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Self-attention layer for object interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # dropout=dropout_p/2,
            dropout=0.0,
            batch_first=True,
        )
        # Processing after attention with normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.post_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, object_features, object_types):
        batch_size, num_objects, feature_dim = x.shape
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            object_features[...,:4] = self.invariant_features(object_features[...,:4])
        combined = torch.cat([object_features, type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Store original features for residual connection
        identity = object_features
        # Apply self-attention to model interactions between objects
        # This creates a mechanism for objects to attend to each other
        attention_output, _ = self.self_attention(
            object_features, object_features, object_features,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Add residual connection and normalize
        attention_output = identity + attention_output
        attention_output = self.layer_norm(attention_output)
        # Post-attention processing
        attention_output = self.post_attention(attention_output)
        return self.classifier(attention_output)

class DeepSetsWithSelfAttention(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32):
        super().__init__()
        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        # Object type embedding
        self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
        # Initial per-object processing
        self.object_net = nn.Sequential(
            nn.Linear(input_dim + embedding_size, hidden_dim),  # All features except type + type embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Self-attention layer for object interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # dropout=dropout_p/2,
            dropout=0.0,
            batch_first=True,
        )
        # Processing after attention
        self.post_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, object_features, object_types):
        batch_size, num_objects, feature_dim = x.shape
        # Get type embeddings and combine with features
        type_emb = self.type_embedding(object_types)
        if USE_LORENTZ_INVARIANT_FEATURES:
            object_features[...,:4] = self.invariant_features(object_features[...,:4])
        combined = torch.cat([object_features, type_emb], dim=-1)
        # Process each object
        object_features = self.object_net(combined)
        # Apply self-attention to model interactions between objects
        # This creates a mechanism for objects to attend to each other
        attention_output, _ = self.self_attention(
            object_features, object_features, object_features,
            key_padding_mask=(object_types==(N_CTX-1))
        )
        # Post-attention processing
        object_features = self.post_attention(attention_output)
        return self.classifier(object_features)



class MyHookedTransformer(HookedTransformer):
    def __init__(self, cfg, is_categorical, num_classes=0, mass_input_layer=2, mass_hidden_dim=256, **kwargs):
        super(MyHookedTransformer, self).__init__(cfg, **kwargs)
        if USE_LORENTZ_INVARIANT_FEATURES:
            self.invariant_features = LorentzInvariantFeatures()
        self.hook_dict['hook_ObjectInputs'] = HookPoint()
        self.hook_dict['hook_ObjectInputs'].name = 'hook_ObjectInputs'
        self.mod_dict['hook_ObjectInputs'] = self.hook_dict['hook_ObjectInputs']
        self.W_Embed = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_Embed, std=0.02)
        self.is_categorical = is_categorical
        if self.is_categorical:
            self.num_classes = num_classes
        self.classifier = nn.Linear(cfg.d_model, num_classes)
        
    def forward(self, tokens: Float[Tensor, "batch object d_input"], token_types: Float[Tensor, "batch object"], **kwargs) -> Float[Tensor, "batch d_model"]:
        self.hook_dict['hook_ObjectInputs'](tokens)
        if USE_LORENTZ_INVARIANT_FEATURES:
            tokens[...,:4] = self.invariant_features(tokens[...,:4])
        expanded_W_E = self.W_Embed.unsqueeze(0).expand(token_types.shape[0], -1, -1, -1)
        expanded_types = token_types.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.W_Embed.shape[-2], self.W_Embed.shape[-1])
        W_E_selected = torch.gather(expanded_W_E, dim=1, index=expanded_types)
        output = einops.einsum(tokens, W_E_selected, "batch object d_input, batch object d_input d_model -> batch object d_model")
        if 'start_at_layer' in kwargs:
            raise NotImplementedError
        else:
            if self.is_categorical:
                class_outs = super(MyHookedTransformer, self).forward(output, start_at_layer=0, stop_at_layer=-1, **kwargs)
                return self.classifier(class_outs)
            else:
                class_outs = super(MyHookedTransformer, self).forward(output, start_at_layer=0, stop_at_layer=-1, **kwargs)
                return class_outs


