# %% Add path to local files in case we're running in a different directory
import sys
sys.path.insert(0,'/users/baines/Code/ChargedHiggs_ProcessingForIntNote/')
# %% [markdown]
# # Load required modules
import time
ts = []
ts.append(time.time())

import numpy as np
from utils.utils import read_file
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from jaxtyping import Float, Int
import einops
import matplotlib.pyplot as plt
from utils.utils import get_num_btags, check_category, Get_PtEtaPhiM_fromXYZT, GetXYZT_FromPtEtaPhiM, GetXYZT_FromPtEtaPhiE, Rotate4VectorPhi, Rotate4VectorEta, Rotate4VectorPhiEta
from torch import nn
# import datasets

# %% Some basic setup
# Some choices about the  process
APPLY_LOW_LEVEL = True
APPLY_HIGH_LEVEL = True
USE_ONE_NET = False
ONLY_CORRECT_TRUTH_FOR_TRAINING = False # Only include the correct truth, for training low-level classification networks with *just* lvbb or qqbb reco events
IS_CATEGORICAL = True
PHI_ROTATED = False
INCLUDE_TAG_INFO = True
USE_LORENTZ_INVARIANT_FEATURES = True
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here or not
CONVERT_TO_PT_PHI_ETA_M = False
MET_CUT_ON = True
REQUIRE_XBB = False # If we only select categories 0, 3, 8, 9, 10 (ie, if INCLUDE_ALL_SELECTIONS is False) then I think this is satisfied anyway
IS_XBB_TAGGED = False
assert (~((REQUIRE_XBB and (~IS_XBB_TAGGED))))
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
if IS_XBB_TAGGED:
    N_CTX = 7 # the SIX types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root files we're reading in.
else:
    N_CTX = 6 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root files we're reading in.
BIN_WRITE_TYPE=np.float32
max_n_objs_for_pred = 15
OUTPUT_DIR = '/data/atlas/baines/20250327v1_Root/'
# OUTPUT_DIR = './tmp/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
if INCLUDE_TAG_INFO:
    N_Real_Vars=5 # px, py, pz, energy, tagInfo, recoInclusion.  BE CAREFUL because this might change and if it does you ahve to rebinarise
else:
    N_Real_Vars=4 # px, py, pz, energy, trueInclusion.  BE CAREFUL because this might change and if it does you ahve to rebinarise
INCLUDE_INCLUSION_TAGS = True # This is only for newer files which contain these tags
if INCLUDE_INCLUSION_TAGS:
    N_Real_Vars += 2
# Create a mapping from the dsid/decay-type pair to integers, for the purposes of binarising data.
dsid_set = np.array([363355,363356,363357,363358,363359,363360,363489,407342,407343,407344,
            407348,407349,410470,410646,410647,410654,410655,411073,411074,411075,
            411077,411078,412043,413008,413023,510115,510116,510117,510118,510119,
            510120,510121,510122,510123,510124,700320,700321,700322,700323,700324,
            700325,700326,700327,700328,700329,700330,700331,700332,700333,700334,
            700338,700339,700340,700341,700342,700343,700344,700345,700346,700347,
            700348,700349,
            ]) # Try not to change this often - have to re-binarise if we do!
DSID_MASS_MAPPING = {510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
MASS_DSID_MAPPING = {v: k for k, v in DSID_MASS_MAPPING.items()} # Create inverse dictionary
# dsid_set = np.array([410470, 510117, 510123])
types_set = np.array([-2, 1, 2])  # Try not to change this often - have to re-binarise if we do!


from utils.utils import check_valid, check_category


















































class LorentzInvariantFeatures(nn.Module):
    def __init__(self):
        super().__init__()
    
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
        
        return torch.stack([mass_squared, pt, eta, phi], dim=-1)

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
        batch_size, num_objects, feature_dim = object_features.shape
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
class DeepSetsWithResidualSelfAttentionVariableTrueSkipReco(nn.Module):
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
                identity = residual
                # normed_mlpin = block['layer_norm2'](residual)
                # Post-attention processing
                mlp_output = block['post_attention'](residual)
                object_features = identity + mlp_output
            else:
                object_features = identity + attention_output
        return self.classifier(object_features)


class DeepSetsWithResidualSelfAttentionVariableTrueSkipClass(nn.Module):
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
                    dropout=0.0,
                    batch_first=True,
                ),
                'layer_norm2': nn.LayerNorm(hidden_dim),
                **({'post_attention': nn.Sequential(
                    # nn.Linear(hidden_dim, hidden_dim_mlp),
                    # nn.GELU(),
                    # nn.Dropout(dropout_p),
                    # nn.Linear(hidden_dim_mlp, hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
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
                identity = residual
                normed_mlpin = block['layer_norm2'](residual)
                # Post-attention processing
                mlp_output = block['post_attention'](normed_mlpin)
                object_features = identity + mlp_output
            else:
                object_features = identity + attention_output

        # Pool by taking mean of non-padding to ensure permutation invariance
        pooled = torch.sum(object_features, dim=1) / torch.sum(object_types!=(N_CTX-1), dim=-1).unsqueeze(-1)
        return self.classifier(pooled)

class DeepSetsWithResidualSelfAttentionVariableTrueSkipClassModified(nn.Module):
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

class ConfigurableNN(nn.Module): # For high level
    def __init__(self, N_inputs, N_targets, hidden_layers, dropout_prob=0.0, use_batchnorm=True):
        """
        Initializes a configurable neural network.
        Args:
            N_inputs (int): Number of input features.
            N_targets (int): Number of output neurons (for classification).
            hidden_layers (list of int): List of integers specifying the number of neurons in each hidden layer.
            dropout_prob (float): Probability of dropout. 0.0 means no dropout.
            use_batchnorm (bool): Whether to use batch normalization in each layer.
        """
        super(ConfigurableNN, self).__init__()
        layers = []
        self.N_inputs = N_inputs
        self.N_targets = N_targets
        input_dim = self.N_inputs
        for hidden_dim in hidden_layers:
            # Input to first hidden layer and subsequent hidden layers
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())  # Activation function
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(p=dropout_prob))
            input_dim = hidden_dim
        # Output layer
        layers.append(nn.Linear(input_dim, N_targets))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def summary(self):
        """
        Prints the summary of the network architecture.
        """
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Network Summary:\n{'-'*20}")
        print(f"Total Parameters: {num_params}")
        print(f"Input size: {self.N_inputs}")
        print(f"Output size: {self.N_targets}")
        print(f"Layers:")
        for i, layer in enumerate(self.network):
            print(f"Layer {i+1}: {layer}")
        print(f"{'-'*20}")


# %%
# Create a new model
if IS_CATEGORICAL:
    num_classes=3
else:
    num_classes=0
device = 'cuda'
model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":10, "num_heads":4}
if USE_ONE_NET:
    model = DeepSetsWithResidualSelfAttentionTriple(num_classes=num_classes, input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
else:
    models = {
        0:DeepSetsWithResidualSelfAttentionVariableTrueSkipReco(num_attention_blocks=3, include_mlp=False, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=200,  dropout_p=0.1,  num_heads=4, embedding_size=10).to(device),
        1:DeepSetsWithResidualSelfAttentionVariableTrueSkipReco(num_attention_blocks=3, include_mlp=False, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=200,  dropout_p=0.1,  num_heads=4, embedding_size=10).to(device),
    }
    models_class = {
        'lvbb':{
            0:DeepSetsWithResidualSelfAttentionVariableTrueSkipClassModified(num_attention_blocks=3, include_mlp=True, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=256, hidden_dim_mlp=256, dropout_p=0.0,  num_heads=4, embedding_size=16).to(device),
            1:DeepSetsWithResidualSelfAttentionVariableTrueSkipClass(num_attention_blocks=3, include_mlp=True, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=256, hidden_dim_mlp=256, dropout_p=0.0,  num_heads=2, embedding_size=10).to(device),
        },
        'qqbb':{
            0:DeepSetsWithResidualSelfAttentionVariableTrueSkipClass(num_attention_blocks=3, include_mlp=True, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=256, hidden_dim_mlp=256, dropout_p=0.0,  num_heads=2, embedding_size=10).to(device),
            1:DeepSetsWithResidualSelfAttentionVariableTrueSkipClass(num_attention_blocks=3, include_mlp=True, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=256, hidden_dim_mlp=256, dropout_p=0.0,  num_heads=2, embedding_size=10).to(device),
        }
    }

if 1: # 
    print("WARNING: You are starting from a semi-pre-trained model state")
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250302-224613_TrainingOutput/models/0/chkpt74_414975.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250304-145036_TrainingOutput/models/0/chkpt163050.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250305-195515_TrainingOutput/models/0/chkpt163050.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250305-232917_TrainingOutput/models/0/chkpt162930.pth" # 
    if APPLY_LOW_LEVEL:
        if USE_ONE_NET:
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250307-191542_TrainingOutput/models/0/chkpt162930.pth" # DSSAR3 d_model=256, n_heads=4, num_embedding=10,
            loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
            model.load_state_dict(loaded_state_dict)
            modeltag = modelfile.split('/')[-4].split('_')[0]
        else:
            modelfile0="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250321-124811_TrainingOutput/models/Nplits2_ValIdx0/chkpt4_27415.pth" # DSSAR3 d_model=256, n_heads=4, num_embedding=10,
            loaded_state_dict = torch.load(modelfile0, map_location=torch.device(device))
            models[0].load_state_dict(loaded_state_dict)

            modelfile1="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250321-124953_TrainingOutput/models/Nplits2_ValIdx1/chkpt4_27440.pth" # DSSAR3 d_model=256, n_heads=4, num_embedding=10,
            loaded_state_dict = torch.load(modelfile1, map_location=torch.device(device))
            models[1].load_state_dict(loaded_state_dict)
            
            modelfile_class0_lvbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250322-033930_TrainingOutput/models/lvbb_Nplits2_ValIdx0/chkpt24_330225.pth"
            if 0:
                for _ in range(5):
                    print("WARNING: USING TRAINING MODEL TO PREDICT WHILST WE WAIT FOR THE OTHER MODEL TO TRAIN")
                modelfile_class1_lvbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250322-005359_TrainingOutput/models/lvbb_Nplits2_ValIdx1/chkpt24_332175.pth"
            loaded_state_dict = torch.load(modelfile_class0_lvbb, map_location=torch.device(device))
            models_class['lvbb'][0].load_state_dict(loaded_state_dict)
            
            modelfile_class1_lvbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250322-005359_TrainingOutput/models/lvbb_Nplits2_ValIdx1/chkpt24_332175.pth"
            loaded_state_dict = torch.load(modelfile_class1_lvbb, map_location=torch.device(device))
            models_class['lvbb'][1].load_state_dict(loaded_state_dict)
            
            modelfile_class0_qqbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250322-011505_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt24_229275.pth"
            loaded_state_dict = torch.load(modelfile_class0_qqbb, map_location=torch.device(device))
            models_class['qqbb'][0].load_state_dict(loaded_state_dict)
            
            modelfile_class1_qqbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250322-011528_TrainingOutput/models/qqbb_Nplits2_ValIdx1/chkpt24_229925.pth"
            loaded_state_dict = torch.load(modelfile_class1_qqbb, map_location=torch.device(device))
            models_class['qqbb'][1].load_state_dict(loaded_state_dict)

            modeltag = 'RECO_' + modelfile0.split('/')[-4].split('_')[0] + modelfile1.split('/')[-4].split('_')[0] + '___CLASS_' + modelfile_class0_lvbb.split('/')[-4].split('_')[0] + modelfile_class0_lvbb.split('/')[-4].split('_')[0] + modelfile_class0_qqbb.split('/')[-4].split('_')[0] + modelfile_class0_qqbb.split('/')[-4].split('_')[0]
    if APPLY_HIGH_LEVEL:
        modelfile_class0_lvbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005558_TrainingOutput/models/lvbb_Nplits2_ValIdx0/chkpt349050.pth"
        modelfile_class1_lvbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005624_TrainingOutput/models/lvbb_Nplits2_ValIdx1/chkpt349350.pth"
        stds_lvbb = np.load(f'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005558_TrainingOutput/models/lvbb_Nplits2_ValIdx0/std.npy')
        means_lvbb = np.load(f'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005558_TrainingOutput/models/lvbb_Nplits2_ValIdx0/mean.npy')

        modelfile_class0_qqbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005646_TrainingOutput/models/qqbb_Nplits2_ValIdx0/chkpt67600.pth"
        modelfile_class1_qqbb="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005715_TrainingOutput/models/qqbb_Nplits2_ValIdx1/chkpt68250.pth"
        stds_qqbb = np.load(f'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005646_TrainingOutput/models/qqbb_Nplits2_ValIdx0/std.npy')
        means_qqbb = np.load(f'/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250326-005646_TrainingOutput/models/qqbb_Nplits2_ValIdx0/mean.npy')

        highLevelModelsDate=modelfile_class0_lvbb.split('/')[-4].split('_')[0]
        models_high_level = {
            'lvbb':{
                0:ConfigurableNN(N_inputs=7, N_targets=2, hidden_layers=[200,200,200], dropout_prob=0.1, use_batchnorm=False).to(device),
                1:ConfigurableNN(N_inputs=7, N_targets=2, hidden_layers=[200,200,200], dropout_prob=0.1, use_batchnorm=False).to(device),
            },
            'qqbb':{
                0:ConfigurableNN(N_inputs=7, N_targets=2, hidden_layers=[200,200,200], dropout_prob=0.1, use_batchnorm=False).to(device),
                1:ConfigurableNN(N_inputs=7, N_targets=2, hidden_layers=[200,200,200], dropout_prob=0.1, use_batchnorm=False).to(device),
            }
        }
        models_high_level['lvbb'][0].load_state_dict(torch.load(modelfile_class0_lvbb, map_location=torch.device(device)))
        models_high_level['lvbb'][1].load_state_dict(torch.load(modelfile_class1_lvbb, map_location=torch.device(device)))
        models_high_level['qqbb'][0].load_state_dict(torch.load(modelfile_class0_qqbb, map_location=torch.device(device)))
        models_high_level['qqbb'][1].load_state_dict(torch.load(modelfile_class1_qqbb, map_location=torch.device(device)))











































# %%
# Main function to actually process a single file and return the arrays
def process_single_file(filepath):
    truth_var = 'll_truth_decay_mode'
    mH_var = 'll_best_mH'
    particle_features=['part_px', 'part_py', 'part_pz', 'part_energy']
    if INCLUDE_TAG_INFO:
        particle_features.append('ll_particle_tagInfo')
    if INCLUDE_INCLUSION_TAGS: # this means we're expecting the reco/true inclusion tags for each particle too
        particle_features += ['ll_particle_recoInclusion','ll_particle_trueInclusion']
    particle_features.append('ll_particle_type')
    x_part, x_event, y = read_file(filepath, 
                                max_num_particles=max_n_objs_for_pred, # This should be high enough that you never end up cutting off an object which is present in the truth matching. In reality, this just needs to be 25 for the samples I have
                                particle_features=particle_features,
                                event_level_features=[
                                    'eventWeight',
                                    'selection_category',
                                    'MET',
                                    'll_best_mWH_qqbb',
                                    'll_best_mWH_lvbb',
                                    mH_var,
                                    'mH',
                                    'nTrkTagsOutside',
                                    'TopHeavyFlavorFilterFlag',
                                    'GenFiltHT',
                                    'weight_normalise',
                                    'weight_pileup',
                                    'weight_mc',
                                    'weight_leptonSF',
                                    'weight_bTagSF_DL1r_Continuous',
                                    'weight_jvt',
                                    'luminosity_factor',
                                    'mVH_lvbb',
                                    'mVH_qqbb',
                                    'NN_lvbb_Final_Combined',
                                    'NN_qqbb_Final_Combined',
                                    'deltaR_LH','deltaPhi_HWlep','deltaEta_HWlep','ratio_Hpt_mVH_lvbb','ratio_Wpt_mVH_lvbb','mTop_lepto','LepEnergyFracLep',
                                    'deltaR_LH','deltaPhi_HWhad','deltaEta_HWhad','ratio_Hpt_mVH_qqbb','ratio_Wpt_mVH_qqbb','deltaR_LWhad','LepEnergyFracHad',
                                ],
                                labels=['DSID', truth_var, 'eventNumber', 'truth_W_decay_mode'],
                                new_inputs_labels=True
    )
    type_part = x_part[:,N_Real_Vars,:]
    type_part[(x_part[:,:N_Real_Vars,:]==0).all(axis=1)] = -1
    # Types dict: 0: electron, 1: muon, 2: neutrino, 3: ljet, 4: sjet, 5: XbbTaggedLjet
    lepton_locs = np.where(((type_part==0)|(type_part==1)))[1]
    lep_XYZT = x_part[np.arange(x_part.shape[0]), :4, lepton_locs]
    lep_pt, lep_eta, lep_phi, _ = Get_PtEtaPhiM_fromXYZT(lep_XYZT[:,0], lep_XYZT[:,1], lep_XYZT[:,2], lep_XYZT[:,3])
    # if MET_CUT_ON:
    #     neutrino_locs = np.where(type_part==2)[1]
    #     neutrino_XYZT = x_part[np.arange(x_part.shape[0]), :4, neutrino_locs]
    #     Neutrino_Pt, _, _, _ = Get_PtEtaPhiM_fromXYZT(neutrino_XYZT[:,0], neutrino_XYZT[:,1], neutrino_XYZT[:,2], neutrino_XYZT[:,3])
    lep_phi_expanded = einops.repeat(lep_phi,'b -> b o',o=x_part.shape[-1])
    lep_eta_expanded = einops.repeat(lep_eta,'b -> b o',o=x_part.shape[-1])
    if 0:
        rotatedx, rotatedy, rotatedz, _ = Rotate4VectorPhiEta(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded, 
                        lep_eta_expanded
                        )
        # rotated_lep_x, _, _, _ = Rotate4VectorPhiEta(*GetXYZT_FromPtEtaPhiE(x_event[:,0],x_event[:,1],x_event[:,2],x_event[:,3]), x_event[:,2], x_event[:,1])
        x_part[:,0,:] = rotatedx - einops.repeat(rotated_lep_x, 'b->b o',o=x_part.shape[-1])
        x_part[:,1,:] = rotatedy
        x_part[:,2,:] = rotatedz
        x_part[:,3,:] = x_part[:,3,:]/ einops.repeat(lep_e, 'b->b o',o=x_part.shape[-1])
    elif 0:
        rotatedx, rotatedy, rotatedz, _ = Rotate4VectorPhiEta(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded, 
                        lep_eta_expanded
                        )
        # rotated_lep_x, _, _, _ = Rotate4VectorPhiEta(*GetXYZT_FromPtEtaPhiE(x_event[:,0],x_event[:,1],x_event[:,2],x_event[:,3]), x_event[:,2], x_event[:,1])
        x_part[:,0,:] = rotatedx - einops.repeat(rotated_lep_x, 'b->b o',o=x_part.shape[-1])
        x_part[:,1,:] = rotatedy
        x_part[:,2,:] = rotatedz
    elif 0:
        rotatedx, rotatedy, rotatedz, _ = Rotate4VectorPhiEta(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded, 
                        lep_eta_expanded
                        )
        x_part[:,0,:] = rotatedx
        x_part[:,1,:] = rotatedy
        x_part[:,2,:] = rotatedz
    elif PHI_ROTATED:
        rotatedx, rotatedy, _, _ = Rotate4VectorPhi(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded
                        )
        x_part[:,0,:] = rotatedx
        x_part[:,1,:] = rotatedy
    if CONVERT_TO_PT_PHI_ETA_M:
        pt, eta, phi, m = Get_PtEtaPhiM_fromXYZT(x_part[:,0,:],x_part[:,1,:],x_part[:,2,:],x_part[:,3,:])
        x_part[:,0,:] = pt
        x_part[:,2,:] = eta
        x_part[:,1,:] = phi
        x_part[:,3,:] = m
    if not IS_XBB_TAGGED:
        type_part[type_part==5] = 3 # Remove the Xbb-vs-notXbb distinction
    type_part[type_part==-1] = N_CTX-1 # Set the non-existing particles to be last (dummy) index of embedding tensor
    x_part = np.concatenate(
        [
            einops.rearrange(type_part,'b o -> b 1 o'),
            x_part[:,:N_Real_Vars,:]
        ],
        axis=1
    )

    event_weights = x_event[:,0]
    selection_category = x_event[:,1]
    # mWh_qqbb = x_event[:,3]
    # mWh_lvbb = x_event[:,4]
    mH_ll = x_event[:,5]
    eventNumber = y[:,2]
    mH = x_event[:,6]
    nTrkTagsOutside = x_event[:,7]
    TopHeavyFlavorFilterFlag = x_event[:,8]
    GenFiltHT = x_event[:,9]
    weight_normalise = x_event[:,10]
    weight_pileup = x_event[:,11]
    weight_mc = x_event[:,12]
    weight_leptonSF = x_event[:,13]
    weight_bTagSF_DL1r_Continuous = x_event[:,14]
    weight_jvt = x_event[:,15]
    luminosity_factor = x_event[:,16]
    mWh_lvbb = x_event[:,17]
    mWh_qqbb = x_event[:,18]
    
    if MET_CUT_ON:
        met = x_event[:,2]

    # Remove events with no large-R jets and which don't pass selection category and which have too low Met Pt
    # Get rid of events where we don't keep enough objects to keep all the relevant truth objects
    if not REQUIRE_XBB:
        no_ljet = (((type_part==3)|(type_part==5)).sum(axis=1) == 0)
    else:
        no_ljet = ((type_part==5).sum(axis=1) == 0)
    if MET_CUT_ON:
        low_MET = met < 30e3 # 30GeV Minimum for MET
    else:
        low_MET = np.zeros_like(no_ljet)#.astype(bool)
    
    num_samps = x_part.shape[0]
    outputs = np.zeros((num_samps, x_part.shape[2], 3))
    class_outputs_lvbb_bkg = np.zeros((num_samps, 1))
    class_outputs_lvbb_lvbb = np.zeros((num_samps, 1))
    class_outputs_qqbb_bkg = np.zeros((num_samps, 1))
    class_outputs_qqbb_qqbb = np.zeros((num_samps, 1))
    highlevel_class_outputs_lvbb = np.zeros((num_samps, 1))
    highlevel_class_outputs_qqbb = np.zeros((num_samps, 1))
    valid = np.zeros(num_samps)
    ll_reco_category = np.ones(num_samps)*-3
    predicted_channel = np.zeros(num_samps)
    if APPLY_LOW_LEVEL:
        if (num_samps>0):
            batch_size = 256
            num_batches = (num_samps-1)//batch_size + 1
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    if USE_ONE_NET:
                        outputs_batch = model(torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                    else:
                        # Reco outputs
                        outputs_batch0 = models[0](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                        outputs_batch1 = models[1](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                        # Now get the relevant prediction (from the net it was val set in, NOT the net it was train set in, based on eventNumber)
                        outputs_batch = outputs_batch0 * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==0).to(float).to(device).reshape(-1,1,1) + outputs_batch1 * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==1).to(float).to(device).reshape(-1,1,1)

                        # Class outputs for lvbb channel
                        class_outputs_batch0_lvbb = models_class['lvbb'][0](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                        class_outputs_batch1_lvbb = models_class['lvbb'][1](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                        class_outputs_batch_lvbb = class_outputs_batch0_lvbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==0).to(float).to(device).reshape(-1,1) + class_outputs_batch1_lvbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==1).to(float).to(device).reshape(-1,1)

                        class_outputs_batch0_qqbb = models_class['qqbb'][0](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                        class_outputs_batch1_qqbb = models_class['qqbb'][1](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                        class_outputs_batch_qqbb = class_outputs_batch0_qqbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==0).to(float).to(device).reshape(-1,1) + class_outputs_batch1_qqbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==1).to(float).to(device).reshape(-1,1)
                    if 1: # Need to put the incorrect reco classification to zero (since this was ignored in the loss calculation)
                        class_outputs_batch_lvbb[...,2] = -torch.inf
                        class_outputs_batch_qqbb[...,1] = -torch.inf
                    elif 0:
                        class_outputs_batch_lvbb[...,2] = 0
                        class_outputs_batch_qqbb[...,1] = 0
                        # WHY DOES ADDING A BUFFER HERE HELP??
                        # Need to check if we can do cleverer things with the logits to make a score
                        # Update: it doesn't help, you made a mistake because you were severly sleep deprived; PhDs suck
                    class_outputs_batch_lvbb = class_outputs_batch_lvbb.softmax(dim=-1)
                    class_outputs_batch_qqbb = class_outputs_batch_qqbb.softmax(dim=-1)
                    

                    
                    outputs[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred] = outputs_batch.cpu().numpy()
                    class_outputs_lvbb_bkg[batch_idx*batch_size:(batch_idx+1)*batch_size] = class_outputs_batch_lvbb.cpu().numpy()[:,0].reshape(-1, 1)
                    class_outputs_lvbb_lvbb[batch_idx*batch_size:(batch_idx+1)*batch_size] = class_outputs_batch_lvbb.cpu().numpy()[:,1].reshape(-1, 1)
                    class_outputs_qqbb_bkg[batch_idx*batch_size:(batch_idx+1)*batch_size] = class_outputs_batch_qqbb.cpu().numpy()[:,0].reshape(-1, 1)
                    class_outputs_qqbb_qqbb[batch_idx*batch_size:(batch_idx+1)*batch_size] = class_outputs_batch_qqbb.cpu().numpy()[:,2].reshape(-1, 1)
                    # pred_inclusion[batch_idx*batch_size:(batch_idx+1)*batch_size] = (outputs.argmax(dim=-1)>0).cpu().numpy()
                    valid_batch, predicted_channel_batch = check_valid(torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size,:max_n_objs_for_pred]), outputs_batch.cpu(), N_CTX-1, IS_CATEGORICAL, returnTypes=True)
                    ll_reco_category_batch = check_category(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size,:max_n_objs_for_pred], outputs_batch.argmax(dim=-1).cpu().numpy(), N_CTX-1)
                    ll_reco_category[batch_idx*batch_size:(batch_idx+1)*batch_size] = ll_reco_category_batch
                    valid[batch_idx*batch_size:(batch_idx+1)*batch_size] = valid_batch.cpu().numpy()
                    predicted_channel[batch_idx*batch_size:(batch_idx+1)*batch_size] = predicted_channel_batch.cpu().numpy()
                    # print(f"Processed batch {batch_idx}/{num_batches}")
            not_valid = ~(valid.astype(bool))
            print(f"{y[0,0]}: {not_valid.sum():7d}/{len(not_valid):7d} ({not_valid.sum()/len(not_valid)*100:5.2f})")
            print(f"{y[0,0]}: {(not_valid*event_weights).sum():10.2f}/{event_weights.sum():10.2f} ({(not_valid*event_weights).sum()/event_weights.sum()*100:5.2f})")
        else:
            print("Skipping since num_samps==0")
            not_valid=np.zeros(num_samps).astype(bool)
    if APPLY_HIGH_LEVEL:
        if (num_samps>0):
            batch_size = 2560
            num_batches = (num_samps-1)//batch_size + 1
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    
                    # Class outputs for lvbb channel
                    class_outputs_batch0_lvbb = models_high_level['lvbb'][0](((torch.Tensor(x_event[batch_idx*batch_size:(batch_idx+1)*batch_size, 21:28]).to(device)-torch.tensor(means_lvbb).to(device))/torch.tensor(stds_lvbb).to(device)).to(torch.float32))
                    class_outputs_batch1_lvbb = models_high_level['lvbb'][1](((torch.Tensor(x_event[batch_idx*batch_size:(batch_idx+1)*batch_size, 21:28]).to(device)-torch.tensor(means_lvbb).to(device))/torch.tensor(stds_lvbb).to(device)).to(torch.float32))

                    class_outputs_batch0_qqbb = models_high_level['qqbb'][0](((torch.Tensor(x_event[batch_idx*batch_size:(batch_idx+1)*batch_size, 28:35]).to(device)-torch.tensor(means_qqbb).to(device))/torch.tensor(stds_qqbb).to(device)).to(torch.float32))
                    class_outputs_batch1_qqbb = models_high_level['qqbb'][1](((torch.Tensor(x_event[batch_idx*batch_size:(batch_idx+1)*batch_size, 28:35]).to(device)-torch.tensor(means_qqbb).to(device))/torch.tensor(stds_qqbb).to(device)).to(torch.float32))

                    class_outputs_batch_lvbb = class_outputs_batch0_lvbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==0).to(float).to(device).reshape(-1,1) + class_outputs_batch1_lvbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==1).to(float).to(device).reshape(-1,1)
                    class_outputs_batch_qqbb = class_outputs_batch0_qqbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==0).to(float).to(device).reshape(-1,1) + class_outputs_batch1_qqbb * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,]).to(int)%2)==1).to(float).to(device).reshape(-1,1)
                    
                    # class_outputs_batch_lvbb = class_outputs_batch_lvbb.sigmoid()
                    # class_outputs_batch_qqbb = class_outputs_batch_qqbb.sigmoid()
                    class_outputs_batch_lvbb = class_outputs_batch_lvbb.softmax(dim=-1)
                    class_outputs_batch_qqbb = class_outputs_batch_qqbb.softmax(dim=-1)
                    

                    highlevel_class_outputs_lvbb[batch_idx*batch_size:(batch_idx+1)*batch_size] = class_outputs_batch_lvbb.cpu().numpy()[:,1:]
                    highlevel_class_outputs_qqbb[batch_idx*batch_size:(batch_idx+1)*batch_size] = class_outputs_batch_qqbb.cpu().numpy()[:,1:]
        else:
            print("Skipping since num_samps==0")
            not_valid=np.zeros(num_samps).astype(bool)

    if APPLY_LOW_LEVEL:
        # Get H+ mass
        fourmom=(einops.rearrange(x_part, 'b v o-> b o v')[:, :max_n_objs_for_pred, 1:4+1] * einops.rearrange((outputs.argmax(axis=-1)>0)[:, :max_n_objs_for_pred], 'b o -> b o 1'))
        _,_,_,mWh_ll=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(axis=-1),fourmom[...,1].sum(axis=-1),fourmom[...,2].sum(axis=-1),fourmom[...,3].sum(axis=-1))
        
        # Get H mass
        fourmom=(einops.rearrange(x_part, 'b v o-> b o v')[:, :max_n_objs_for_pred, 1:4+1] * einops.rearrange((outputs.argmax(axis=-1)==1)[:, :max_n_objs_for_pred], 'b o -> b o 1'))
        _,_,_,mH_ll=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(axis=-1),fourmom[...,1].sum(axis=-1),fourmom[...,2].sum(axis=-1),fourmom[...,3].sum(axis=-1))

    if len(y):
        dsid = y[0,0]
    else:
        dsid = 0 # Let the function run with empty arrays anyway
    if APPLY_LOW_LEVEL:
        # for target_channel, channel_num in [('lvbb', 1), ('qqbb', 2)]:
        reco_selection_lvbb = (predicted_channel==1) & ~(not_valid | no_ljet | low_MET)
        reco_selection_qqbb = (predicted_channel==2) & ~(not_valid | no_ljet | low_MET)
        pred_inclusion = outputs.argmax(axis=-1)
    

    # HERE Need to write some code to store this properly
    # dsid = np.searchsorted(dsid_set, y[:, 0])
    dsid = y[:, 0]
    truth_label = (y[:, 1] == 1)*1 + (y[:, 1] == 2)*2
    # truth_label = np.searchsorted(types_set, y[:, 1])

    # Reshape the x array to how we want to read it in later
    x_part = einops.rearrange(x_part, 'batch d_input object -> batch object d_input')

    if APPLY_LOW_LEVEL:
        # Get the number of b-tags outside the W/Higgs from H+ decay
        assert(INCLUDE_TAG_INFO) # Need this to be on for the tag inclusion
        n_btags_ll = np.ones_like(truth_label)*-1
        n_btags_ll[reco_selection_lvbb] = get_num_btags(x_part[reco_selection_lvbb, :, 1:6], type_part[reco_selection_lvbb], pred_inclusion[reco_selection_lvbb], 4, use_torch=False)
        n_btags_ll[reco_selection_qqbb] = get_num_btags(x_part[reco_selection_qqbb, :, 1:6], type_part[reco_selection_qqbb], pred_inclusion[reco_selection_qqbb], 4, use_torch=False)
    
    if 0:
        return truth_label, dsid, event_weights, removals, mWh_qqbb, mWh_lvbb, mH_ll, eventNumber, class_outputs_lvbb_bkg, class_outputs_lvbb_lvbb, class_outputs_qqbb_bkg, class_outputs_qqbb_qqbb, n_btags_ll, reco_selection_lvbb, reco_selection_qqbb
    else:
        ll_reco_selection = reco_selection_lvbb.astype(int)*1 + reco_selection_qqbb.astype(int)*2
        category = np.ones_like(selection_category)
        category[selection_category==0] = 0
        category[selection_category==8] = 0
        category[selection_category==10] = 0
        category[selection_category==3] = 3
        category[selection_category==9] = 3
        mWh = mWh_lvbb*(category==0) + mWh_qqbb*(category==3)
        return {
            'dsid': dsid,
            'weights': event_weights,
            'eventNumber': eventNumber,
            'll_truth_label': truth_label,
            'll_mWh': mWh_ll,
            'll_mH': mH_ll,
            'll_class_outputs_lvbb_bkg': class_outputs_lvbb_bkg,
            'll_class_outputs_lvbb_lvbb': class_outputs_lvbb_lvbb,
            'll_class_outputs_qqbb_bkg': class_outputs_qqbb_bkg,
            'll_class_outputs_qqbb_qqbb': class_outputs_qqbb_qqbb,
            'll_numbtags': n_btags_ll,
            'll_reco_selection': ll_reco_selection,
            'll_reco_category':ll_reco_category,
            # Old variables
            'mH':mH,
            'nTrkTagsOutside':nTrkTagsOutside,
            'mWh':mWh,
            'selection_category':selection_category,
            'category':category,
            'NN_lvbb_Final_Combined':x_event[:,19],
            'NN_qqbb_Final_Combined':x_event[:,20],
            f'NN_lvbb_Final_Combined_AllSigTrainedBest_{highLevelModelsDate}':highlevel_class_outputs_lvbb,
            f'NN_qqbb_Final_Combined_AllSigTrainedBest_{highLevelModelsDate}':highlevel_class_outputs_qqbb,
            # Weight calculation for in case we want to use different filtered samples
            'TopHeavyFlavorFilterFlag':TopHeavyFlavorFilterFlag,
            'GenFiltHT':GenFiltHT,
            'weight_normalise':weight_normalise,
            'weight_pileup':weight_pileup,
            'weight_mc':weight_mc,
            'weight_leptonSF':weight_leptonSF,
            'weight_bTagSF_DL1r_Continuous':weight_bTagSF_DL1r_Continuous,
            'weight_jvt':weight_jvt,
            'luminosity_factor':luminosity_factor,
            'truth_W_decay_mode':y[:,3],
        }

def combine_arrays_for_writing(x_chunk, y_chunk, dsid_chunk, weights_chunk, mWh_qqbbs_chunk, mWh_lvbbs_chunk, mH_chunk, eventNumber_chunk):
    # print(type(y_chunk))
    # print(y_chunk.shape)
    y_chunk = einops.repeat(y_chunk,'b -> b 1 nvars', nvars=x_chunk.shape[-1]).astype(BIN_WRITE_TYPE)
    # print(type(y_chunk))
    # print(y_chunk.shape)
    y_chunk[:, 0, 1] = mH_chunk.squeeze()
    y_chunk[:, 0, 2] = mWh_qqbbs_chunk.squeeze()
    y_chunk[:, 0, 3] = mWh_lvbbs_chunk.squeeze()
    extra_info_chunk = np.zeros_like(y_chunk)
    extra_info_chunk[:, 0, 0] = weights_chunk.squeeze()
    extra_info_chunk[:, 0, 1] = dsid_chunk.squeeze()
    extra_info_chunk[:, 0, 2] = eventNumber_chunk.squeeze()
    array_to_write=np.float32(np.concatenate(
        [
            y_chunk,
            extra_info_chunk,
            x_chunk
        ],
    axis=-2
    ))
    # np.random.shuffle(array_to_write)
    return array_to_write


# %%
types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'Xbb_ljet'}
DATA_PATH='/data/atlas/HplusWh/20250313_WithTrueInclusion_FixedOverlapWHsjet_SmallJetCloseToLargeJetRemovalDeltaR0.5/'
MAX_CHUNK_SIZE = 100000
import uproot
import awkward as ak
for dsid in dsid_set:
    for channel in ['lvbb']:
        # if (dsid < 500000) or (dsid > 600000): # Is background
        # if (dsid < 500000) or (dsid > 600000): # Is background
        # if (dsid > 500000) and (dsid < 600000): # Is signal
        # if dsid > 400000:
        # if dsid != 510120:
        #     continue
        
        all_files = []
        nfs = 0
        removals = {0:0, 1:0}
        for filename in os.listdir(DATA_PATH):
            if (str(dsid) in filename) and (filename.endswith('.root') and (filename.startswith('user'))):
                all_files.append(DATA_PATH + '/' + filename)
        output_path = os.path.join(OUTPUT_DIR, f'dsid_{dsid}_{channel}.root')
        with uproot.recreate(output_path) as f:
            # Initialise tree to None, will make when we know what the branch names will be
            tree = None
            total_events = 0
            sum_abs_weights = 0
            sum_weights = 0
            for file_n, path in enumerate(all_files):
                data = process_single_file(filepath=path)
                if len(data['weights']) > 0:
                    if tree is None:
                        # Create tree
                        branches = {
                            'dsid': "int64",
                            'weights': "float32",
                            'eventNumber': "int64",
                            'll_truth_label': "int64",
                            'll_mWh': "float32",
                            'll_mH': "float32",
                            'll_class_outputs_lvbb_bkg': "float32",
                            'll_class_outputs_lvbb_lvbb': "float32",
                            'll_class_outputs_qqbb_bkg': "float32",
                            'll_class_outputs_qqbb_qqbb': "float32",
                            'll_numbtags':"int64",
                            'll_reco_selection':"int64",
                            'll_reco_category':"int64",
                            # Old variables
                            'mH':"float32",
                            'nTrkTagsOutside':"int64",
                            'mWh':"float32",
                            'selection_category':"int64",
                            'category':"int64",
                            'NN_lvbb_Final_Combined':"float32",
                            'NN_qqbb_Final_Combined':"float32",
                            'truth_W_decay_mode':'int64',
                            # Weight calculation for in case we want to use different filtered samples
                            'TopHeavyFlavorFilterFlag':"int64",
                            'GenFiltHT':"float32",
                            'weight_normalise':"float32",
                            'weight_pileup':"float32",
                            'weight_mc':"float32",
                            'weight_leptonSF':"float32",
                            'weight_bTagSF_DL1r_Continuous':"float32",
                            'weight_jvt':"float32",
                            'luminosity_factor':"float32",
                        }
                        # Any other branches, try and add them as float32
                        branches.update({k:"float32" for k in data.keys() if k not in branches})
                        tree = f.mktree("Events", branches)
                    tree.extend({k:data[k].flatten() for k in data})
                    total_events += len(data['weights'])
                    sum_abs_weights += np.abs(data['weights']).sum()
                    sum_weights += data['weights'].sum()
            
            nfs+=1
            if nfs > 1000000:
                break
            # Write metadata
            # f["metadata"] = {
            #     "total_events": total_events,
            #     "sum_abs_weights": sum_abs_weights,
            #     "sum_weights": sum_weights
            # }
        print("Total accepted %s: %d" %(channel, total_events))
        print("Total bkg removed for no ljet/MET Cut/Selection category fail: %d" %(removals[0]))
        print("Total sig removed for no ljet/MET Cut/Selection category fail: %d" %(removals[1]))

# %%
