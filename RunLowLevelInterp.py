# %% # Load required modules
from interp import mechinterputils
import importlib 
import numpy as np
import os
import torch
from datetime import datetime
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from torch import nn
import matplotlib.pyplot as plt
import shutil
from metrics.lowlevelmetrics import HEPMetrics
from models.models import TestNetwork

# %%
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")
# saveDir = "output/" + timeStr  + "_MechInterpOutput/"
# os.makedirs(saveDir)
# print(saveDir)


# Some choices about the training process
# Assumes that the data has already been binarised
ONLY_SIG = True # Have this as true if we are only wanting to look at signal samples. In general this will make processing faster/mean we have to cut out less of the samples if we are ONLY looking to do mech interp on signal. Sometimes this will be the case, but be warned because sometimes we will want to understand what it's doing on bkg samples too
IS_CATEGORICAL = True
PHI_ROTATED = False
REMOVE_WHERE_TRUTH_WOULD_BE_CUT = True
MODEL_ARCH="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP"
device = "cpu"
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
TOSS_UNCERTAIN_TRUTH = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = False
SHUFFLE_OBJECTS = False
NORMALISE_DATA = False
SCALE_DATA = True
assert(not (NORMALISE_DATA and SCALE_DATA))
CONVERT_TO_PT_PHI_ETA_M = False
IS_XBB_TAGGED = False
REQUIRE_XBB = False # If we only select categories 0, 3, 8, 9, 10 (ie, if INCLUDE_ALL_SELECTIONS is False) then I think this is satisfied anyway
assert (~((REQUIRE_XBB and (~IS_XBB_TAGGED))))
INCLUDE_TAG_INFO = True
INCLUDE_ALL_SELECTIONS = True
INCLUDE_NEGATIVE_SELECTIONS = True
if (INCLUDE_NEGATIVE_SELECTIONS and (not INCLUDE_ALL_SELECTIONS)):
    assert(False)
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
if IS_XBB_TAGGED:
    N_CTX = 7 # the SIX types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
else:
    N_CTX = 6 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet'}
padding_token = N_CTX-1
USE_DROPOUT = True
BIN_WRITE_TYPE=np.float32
max_n_objs_in_file = 15 # BE CAREFUL because this might change and if it does you ahve to rebinarise
max_n_objs_to_read = 15
assert(max_n_objs_in_file==max_n_objs_to_read) # Otherwise we might cut off truth info
assert(REMOVE_WHERE_TRUTH_WOULD_BE_CUT) # Otherwise we might have already cut off truth info in writing the file
if INCLUDE_TAG_INFO:
    N_Real_Vars=5 # px, py, pz, energy, tagInfo.  BE CAREFUL because this might change and if it does you ahve to rebinarise
    N_Real_Vars_InFile = 7
else:
    N_Real_Vars=4 # px, py, pz, energy.  BE CAREFUL because this might change and if it does you ahve to rebinarise
    N_Real_Vars_InFile = 6 # Plus naive-reco-inclusion, true-inclusion

# %%
# Set up stuff to read in data from bin file
batch_size = 3000
# This first one has removed events that aren't validly reconstructed (which might be interesting to look at)
DATA_PATH='/data/atlas/baines/20250314v1_AppliedRecoNN_WithRecoMasses_30_MetCut_RemovedUncertainTruth_WithTagInfo_KeepAllOldSelIncludingNegative/'
# This one doesn't contain all the background as far as I can tell. Might need to add this at some point
DATA_PATH=f'/data/atlas/baines/20250313v1_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'
DATA_PATH=f'/data/atlas/baines/20250321v1_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'

if 'AppliedRecoNN' in DATA_PATH:
    N_Real_Vars_InFile+=1 # Plus NN-reco-inclusion INBETWEEN THE naive-reco-inclusion and true-inclusion

if NORMALISE_DATA:
    means = np.load(f'{DATA_PATH}mean.npy')[1:]
    stds = np.load(f'{DATA_PATH}std.npy')[1:]
else:
    means = None
    if SCALE_DATA:
        stds = np.ones(N_Real_Vars_InFile)
        stds[:4] = 1e5
    else:
        stds = None
memmap_paths_train = {}
memmap_paths_val = {}
for file_name in os.listdir(DATA_PATH):
    if ('shape' in file_name) or ('npy' in file_name):
        continue
    dsid = file_name[5:11]
    if (int(dsid) > 500000) and (int(dsid) < 600000):
        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    else:
        pass # Skip because we don't train the reco on bkg
    if ((int(dsid) > 500000) and (int(dsid) < 600000)) or (not (ONLY_SIG)):
    # if True:
        # if int(dsid)!=510115:
        #     continue
        memmap_paths_val[int(dsid)] = DATA_PATH+file_name
    else:
        print("For now, don't include the background even in the val set, but later we might want to add some backgorund performance logging to the Metric tracking code")
n_splits=2
validation_split_idx=0
val_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_val,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars_InFile,
                 N_Real_Vars_To_Return=N_Real_Vars_InFile,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device,
                 is_train=False,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 shuffle_batch=False,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 signal_only=True,
                #  has_eventNumbers=True,
)
print(val_dataloader.get_total_samples())

# %%


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

if MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X3":
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
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE":
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
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":
    class DeepSetsWithResidualSelfAttentionVariableTrueSkip(nn.Module):
        def __init__(self, input_dim=None, feature_set=['pt', 'eta', 'phi', 'm', 'tag'], num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None, use_lorentz_invariant_features=True):
            super().__init__()
            if input_dim is not None:
                print("WARNING: input_dim is not used in this model, using feature_set instead")
            self.feature_set = feature_set
            self.num_attention_blocks = num_attention_blocks
            self.include_mlp = include_mlp
            self.use_lorentz_invariant_features = use_lorentz_invariant_features
            if hidden_dim_mlp is None:
                hidden_dim_mlp = hidden_dim

            if self.use_lorentz_invariant_features:
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
            if self.use_lorentz_invariant_features:
                invariant_features = self.invariant_features(object_features[...,:4])
            else:
                invariant_features = object_features[...,:4]
            features_to_stack = [invariant_features]
            if 'tag' in self.feature_set:
                features_to_stack.append(object_features[...,4:])
            features_to_stack.append(type_emb)
            combined = torch.cat(features_to_stack, dim=-1)
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
    
    class DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(nn.Module):
        def __init__(self, bottleneck_attention=None, feature_set=['pt', 'eta', 'phi', 'm', 'tag'], use_lorentz_invariant_features=True, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None):
            super().__init__()
            self.bottleneck_attention = bottleneck_attention
            self.num_attention_blocks = num_attention_blocks
            self.include_mlp = include_mlp
            self.use_lorentz_invariant_features = use_lorentz_invariant_features
            if hidden_dim_mlp is None:
                hidden_dim_mlp = hidden_dim

            if use_lorentz_invariant_features:
                self.invariant_features = LorentzInvariantFeatures(feature_set=feature_set)
            
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            
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
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP_OLD":
    class DeepSetsWithResidualSelfAttentionVariableTrueSkip(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True):
            super().__init__()
            self.num_attention_blocks = num_attention_blocks
            self.include_mlp = include_mlp

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
                    # 'layer_norm1': nn.LayerNorm(hidden_dim),
                    'self_attention': nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=0.0,
                        batch_first=True,
                    ),
                    # 'layer_norm2': nn.LayerNorm(hidden_dim),
                    **({'post_attention': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_p),
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
else:
    assert(False)

# %%
# Create a new model
if IS_CATEGORICAL:
    num_classes=3
else:
    num_classes=1
labels = ['Bkg', 'Lep', 'Had'] # truth==0 is bkg, truth==1 is leptonic decay, truth==2 is hadronic decay




# %%
if 1: # Load a pre-trained model
    HAS_MLP = False # Default, to be overwritten by models which do have MLP layers after self-attention
    EXCLUDE_TAG = False  # Default, to be overwritten by models which need fewer inputs
    print("WARNING: You are starting from a semi-pre-trained model state")
    if MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X3":
        # modelfile="model.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
        assert(False) # Need to go find this model
        model = DeepSetsWithResidualSelfAttentionTriple(num_classes=num_classes, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
    elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE":
        if 0:
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250313-162852_TrainingOutput/models/0/chkpt29_164550.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.2, "embedding_size":16, "num_heads":4}
            model = DeepSetsWithResidualSelfAttentionVariable(num_attention_blocks=5, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
            HAS_MLP = True
        elif 0:
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250316-111909_TrainingOutput/models/0/chkpt19_109700.pth"
            model_cfg = {'d_model': 152, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariable(num_attention_blocks=5, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
            HAS_MLP = True
    elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":
        if 0:
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250317-194149_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 152, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(num_attention_blocks=5, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
            HAS_MLP = True
        elif 0: # Without MLPs
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-012556_TrainingOutput/models/0/chkpt29_164550.pth"
            model_cfg = {'d_model': 152, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=8, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
        elif 0: # Without MLPs, 1 layer
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-124405_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=1, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
        elif 0: # Without MLPs, 2 layers
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-123425_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=2, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
        elif 0: # Without MLPs, 3 layers
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-123446_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=3, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
        elif 0: # Without MLPs, 4 layers
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-123605_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=4, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
        elif 0: # With MLPs, 3 layers
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250328-113204_TrainingOutput/models/Nplits2_ValIdx0/chkpt14_82245.pth"
            model_cfg = {'include_mlp':True, 'd_model': 100, 'd_mlp': 400, 'num_blocks':3, 'dropout_p': 0.0, "embedding_size":10, "num_heads":4}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=True, num_attention_blocks=3, hidden_dim=100, hidden_dim_mlp=400, num_heads=4, embedding_size=10, num_classes=3, input_dim=N_Real_Vars,  dropout_p=0.1).to(device)
            HAS_MLP = True
        elif 0: # With MLPs, 3 layers, no lorentz invariant transform
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250331-101705_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_147330.pth"
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(use_lorentz_invariant_features=True, include_mlp=True, num_attention_blocks=3, hidden_dim=100, hidden_dim_mlp=400, num_heads=2, embedding_size=10, num_classes=3, input_dim=N_Real_Vars,  dropout_p=0.1).to(device)
            HAS_MLP = True
        elif 0: # With MLPs, 3 layers, no lorentz invariant transform
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250327-151149_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth"
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(use_lorentz_invariant_features=False, include_mlp=True, num_attention_blocks=3, hidden_dim=152, hidden_dim_mlp=256, num_heads=4, embedding_size=10, num_classes=3, input_dim=N_Real_Vars,  dropout_p=0.1).to(device)
            HAS_MLP = True
        elif 0: # WithOUT MLPs, 3 layers, no lorentz invariant transform
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250402-115455_TrainingOutput/models/Nplits2_ValIdx0/chkpt4_27415.pth"
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(use_lorentz_invariant_features=True, include_mlp=False, num_attention_blocks=2, hidden_dim=300, hidden_dim_mlp=0, num_heads=4, embedding_size=10, num_classes=3, input_dim=N_Real_Vars,  dropout_p=0.0).to(device)
        elif 0: # Fun for testing. WithOUT MLPs, 3 layers, with lorentz invariant transform
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250403-114942_TrainingOutput/models/Nplits2_ValIdx0/chkpt19_109660.pth"
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(use_lorentz_invariant_features=True, include_mlp=False, num_attention_blocks=3, hidden_dim=400, hidden_dim_mlp=0, num_heads=2, embedding_size=10, num_classes=3, input_dim=N_Real_Vars,  dropout_p=0.0).to(device)
        elif 0: # Pretty good one. With MLPs, 3 layers, lorentz invariant transform
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250328-113053_TrainingOutput/models/Nplits2_ValIdx0/chkpt24_137075.pth"
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(use_lorentz_invariant_features=True, include_mlp=True, num_attention_blocks=3, hidden_dim=152, hidden_dim_mlp=400, num_heads=4, embedding_size=10, num_classes=3, input_dim=N_Real_Vars,  dropout_p=0.0).to(device)
            HAS_MLP = True
        elif 0: # Only use 4-vector (no btag info)
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250410-124045_TrainingOutput/models/Nplits2_ValIdx0/chkpt14_82245.pth"
            EXCLUDE_TAG = True
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(feature_set=['pt', 'eta', 'phi', 'm'], use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=3, hidden_dim=400, hidden_dim_mlp=1, num_heads=2, embedding_size=10, num_classes=3, input_dim=N_Real_Vars-int(EXCLUDE_TAG),  dropout_p=0.0).to(device)
        elif 0: # Only use angles (no btag info, mass, pt)
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250410-154021_TrainingOutput/models/Nplits2_ValIdx0/chkpt4_27415.pth"
            EXCLUDE_TAG = True
            HAS_MLP=True
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(feature_set=['eta', 'phi'], use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=3, hidden_dim=152, hidden_dim_mlp=300, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
        elif 0: # Only use angles (no btag info, mass, pt)
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250410-162312_TrainingOutput/models/Nplits2_ValIdx0/chkpt0_5483.pth"
            EXCLUDE_TAG = True
            HAS_MLP=False
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(feature_set=['eta', 'phi'], use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=1, hidden_dim=152, hidden_dim_mlp=300, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
        elif 0: # Only use angles (no btag info, mass, pt)
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250410-161056_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_54830.pth"
            EXCLUDE_TAG = True
            HAS_MLP=True
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(feature_set=['eta', 'phi'], use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=1, hidden_dim=300, hidden_dim_mlp=400, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
        elif 0: # Only use angles (no btag info, mass, pt)
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250413-185222_TrainingOutput/models/Nplits2_ValIdx0/chkpt3_21932.pth"
            EXCLUDE_TAG = False
            HAS_MLP=False
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(feature_set=['eta', 'phi', 'pt', 'm', 'tag'], use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=3, hidden_dim=300, hidden_dim_mlp=1, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
        elif 0: # Only use angles (no btag info, mass, pt)
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250413-234444_TrainingOutput/models/Nplits2_ValIdx0/chkpt11_65796.pth"
            EXCLUDE_TAG = True
            HAS_MLP=False
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(bottleneck_attention=1, feature_set=['eta', 'phi', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=3, hidden_dim=300, hidden_dim_mlp=1, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
            fwd_hooks = []
            hooks = []
            dummy_cache = mechinterputils.ActivationCache()
            fwd_hooks.extend(mechinterputils.hook_attention_heads(model, dummy_cache, detach=True, SINGLE_ATTENTION=False, bottleneck_attention_output=model.bottleneck_attention))
            for module, hook_fn in fwd_hooks:
                hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
        elif 0: # Only use angles (no btag info, mass, pt)
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250414-170725_TrainingOutput/models/Nplits2_ValIdx0/chkpt4_27415.pth"
            EXCLUDE_TAG = True
            HAS_MLP=False
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(bottleneck_attention=1, feature_set=['eta', 'phi', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=3, hidden_dim=300, hidden_dim_mlp=1, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
            fwd_hooks = []
            hooks = []
            dummy_cache = mechinterputils.ActivationCache()
            fwd_hooks.extend(mechinterputils.hook_attention_heads(model, dummy_cache, detach=True, SINGLE_ATTENTION=True, min_attention=0.5, bottleneck_attention_output=model.bottleneck_attention))
            for module, hook_fn in fwd_hooks:
                hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
        elif 0: # 
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250414-221359_TrainingOutput/models/Nplits2_ValIdx0/chkpt2_16449.pth"
            EXCLUDE_TAG = False
            HAS_MLP=False
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(bottleneck_attention=1, feature_set=['eta', 'phi', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=3, hidden_dim=300, hidden_dim_mlp=1, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
            fwd_hooks = []
            hooks = []
            dummy_cache = mechinterputils.ActivationCache()
            fwd_hooks.extend(mechinterputils.hook_attention_heads(model, dummy_cache, detach=True, SINGLE_ATTENTION=False, min_attention=None, bottleneck_attention_output=model.bottleneck_attention))
            for module, hook_fn in fwd_hooks:
                hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
        elif 0: # ?Interesting to switch between ~chkpt4 and ~chkpt9 or so because it seems to have a step change in ability
            # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250414-221359_TrainingOutput/models/Nplits2_ValIdx0/chkpt4_27415.pth"
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250414-221359_TrainingOutput/models/Nplits2_ValIdx0/chkpt10_60313.pth"
            # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250414-221359_TrainingOutput/models/Nplits2_ValIdx0/chkpt17_98694.pth"
            EXCLUDE_TAG = False
            HAS_MLP=False
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(bottleneck_attention=1, feature_set=['eta', 'phi', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), use_lorentz_invariant_features=True, include_mlp=HAS_MLP, num_attention_blocks=3, hidden_dim=300, hidden_dim_mlp=1, num_heads=4, embedding_size=10, num_classes=3,  dropout_p=0.0).to(device)
            fwd_hooks = []
            hooks = []
            dummy_cache = mechinterputils.ActivationCache()
            fwd_hooks.extend(mechinterputils.hook_attention_heads(model, dummy_cache, detach=True, SINGLE_ATTENTION=False, min_attention=None, bottleneck_attention_output=model.bottleneck_attention))
            for module, hook_fn in fwd_hooks:
                hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
        elif 1: # Test with low dimensionality network
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250429-104046_TrainingOutput/models/Nplits2_ValIdx0/chkpt19_109660.pth"
            TAG_INFO_INPUT=True
            HAS_MLP = False
            model = TestNetwork(hidden_dim_attn=2, hidden_dim=100, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, bottleneck_attention=None, include_mlp=HAS_MLP, num_attention_blocks=4, hidden_dim_mlp=1, num_heads=2, embedding_size=10, num_classes=3, use_lorentz_invariant_features=True, dropout_p=0.0, num_particle_types=N_CTX).to(device)
    else:
        assert(False)
    loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
    model.load_state_dict(loaded_state_dict)
else: # This is where we would run a training loop
    pass

# %%
# Evaluate the model
if 1:
    val_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
    val_dataloader._reset_indices()
    val_metrics_MCWts.reset()
    model.eval()
    num_batches_to_process = int(300000 * (1/batch_size))
    # num_batches_to_process = len(val_dataloader)
    for batch_idx in range(num_batches_to_process):
        if ((batch_idx%10)==9):
            print(F"Processing batch {batch_idx}/{num_batches_to_process}")
        batch = next(val_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
        if 0: # See what it's like if we randomly guess
            outputs = torch.rand(x.shape[0], x.shape[1], 3)
        # outputs, cache = mechinterputils.run_with_cache_and_bottleneck(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
        outputs = outputs.squeeze()
        val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)

rv=val_metrics_MCWts.compute_and_log(1,'val', 0, 3, False, None, calc_all=True)


# %%
for k in rv.keys():
    if 'tRecoPct_all_cat' in k:
        print(f"{k:20s}: {rv[k]:.4f}")
# category[valid_H_sjet & valid_W_sjet]   = 0
# category[valid_H_sjet & valid_W_lv]     = 1
# category[valid_H_sjet & valid_W_ljet]   = 2
# category[valid_H_ljet & valid_W_sjet]   = 3
# category[valid_H_ljet & valid_W_lv]     = 4
# category[valid_H_ljet & valid_W_ljet]   = 5



# %%
# assert(False)

# %%
# Evaluate the model BUT only allow each object to pay attention to ONE other object (force 'single-attention')
if 0:
    val_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
    val_dataloader._reset_indices()
    val_metrics_MCWts.reset()
    model.eval()
    num_batches_to_process = int(30000 * (1/batch_size))
    # num_batches_to_process = len(val_dataloader)
    for batch_idx in range(num_batches_to_process):
        if ((batch_idx%10)==9):
            print(F"Processing batch {batch_idx}/{num_batches_to_process}")
        batch = next(val_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        # outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types).squeeze()
        # cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
        outputs, cache = mechinterputils.run_with_cache_and_singleAttention(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
        outputs = outputs.squeeze()
        val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)

rvsa=val_metrics_MCWts.compute_and_log(1,'val', 0, 3, False, None)

# %%
# Check the incorrect-reconstruction rate
if 1:
    from utils import check_valid, DSID_MASS_MAPPING
    val_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
    val_dataloader._reset_indices()
    val_metrics_MCWts.reset()
    model.eval()
    num_batches_to_process = int(30000 * (1/batch_size))
    # num_batches_to_process = len(val_dataloader)
    wts = []
    rts = []
    tts = []
    ds = []
    for batch_idx in range(num_batches_to_process):
        if ((batch_idx%10)==9):
            print(F"Processing batch {batch_idx}/{num_batches_to_process}")
        batch = next(val_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types).squeeze()
        _, reco_type = check_valid(types, outputs, N_CTX-1, IS_CATEGORICAL, returnTypes=True)
        truth_type = y.argmax(dim=-1)
        wts.append(w)
        rts.append(reco_type)
        tts.append(truth_type)
        ds.append(dsids)
    wts = np.concatenate(wts)
    rts = np.concatenate(rts)
    tts = np.concatenate(tts)
    ds = np.concatenate(ds)
    
    for dsid in np.unique(ds):
        true_lvbb_reco_lvbb = wts[(tts == 1) & (rts == 1) & (ds == dsid)].sum()
        true_lvbb_reco_qqbb = wts[(tts == 1) & (rts == 2) & (ds == dsid)].sum()
        true_qqbb_reco_lvbb = wts[(tts == 2) & (rts == 1) & (ds == dsid)].sum()
        true_qqbb_reco_qqbb = wts[(tts == 2) & (rts == 2) & (ds == dsid)].sum()
        misreco_rate_lvbb = (true_lvbb_reco_qqbb / (true_lvbb_reco_lvbb + true_lvbb_reco_qqbb)) * 100
        misreco_rate_qqbb = (true_qqbb_reco_lvbb / (true_qqbb_reco_lvbb + true_qqbb_reco_qqbb)) * 100
        misreco_rate_all = ((true_qqbb_reco_lvbb + true_lvbb_reco_qqbb) / (true_qqbb_reco_lvbb + true_qqbb_reco_qqbb + true_lvbb_reco_lvbb + true_lvbb_reco_qqbb)) * 100
        print(f"{DSID_MASS_MAPPING[dsid]}: lvbb mis-reco = {misreco_rate_lvbb:5.3f}, qqbb mis-reco = {misreco_rate_qqbb:5.3f}, all mis-reco = {misreco_rate_all:5.3f}")

# %% # Get the above as percentages to use in plots
true_lvbb_reco_lvbb = np.zeros_like(np.unique(ds))
true_lvbb_reco_qqbb = np.zeros_like(np.unique(ds))
true_qqbb_reco_lvbb = np.zeros_like(np.unique(ds))
true_qqbb_reco_qqbb = np.zeros_like(np.unique(ds))
for n, dsid in enumerate(sorted(np.unique(ds))):
    true_lvbb_reco_lvbb[n] = wts[(tts == 1) & (rts == 1) & (ds == dsid)].sum() / wts[(ds == dsid) & (rts!=0) & (tts!=0)].sum()*100
    true_lvbb_reco_qqbb[n] = wts[(tts == 1) & (rts == 2) & (ds == dsid)].sum() / wts[(ds == dsid) & (rts!=0) & (tts!=0)].sum()*100
    true_qqbb_reco_lvbb[n] = wts[(tts == 2) & (rts == 1) & (ds == dsid)].sum() / wts[(ds == dsid) & (rts!=0) & (tts!=0)].sum()*100
    true_qqbb_reco_qqbb[n] = wts[(tts == 2) & (rts == 2) & (ds == dsid)].sum() / wts[(ds == dsid) & (rts!=0) & (tts!=0)].sum()*100

# %%
# Evaluate the model BUT switch the last small-R jet to a neutrino to see how much it messes stuff up
# Predictably, it now does really badly (esp at lvbb)
if 0:
    val_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
    val_metrics_MCWts2 = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
    val_dataloader._reset_indices()
    val_metrics_MCWts.reset()
    val_metrics_MCWts2.reset()
    model.eval()
    num_batches_to_process = int(30000 * (1/batch_size))
    # num_batches_to_process = len(val_dataloader)
    for batch_idx in range(num_batches_to_process):
        if ((batch_idx%10)==9):
            print(F"Processing batch {batch_idx}/{num_batches_to_process}")
        batch = next(val_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()

        # Make new inputs with the last small-R jet replaced with lepton
        is_sjet = (types==4).to(int)
        last_sjet_index = (is_sjet.size(1) - 1) - torch.argmax(is_sjet.flip(dims=[1]), dim=1)
        last_sjet_index
        types_new = types.clone()
        types_new[torch.arange(types.shape[0]),last_sjet_index] = 2

        outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types_new).squeeze()
        val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types_new)
        val_metrics_MCWts2.update(outputs, x[...,-1], MCWts, dsids, types)

    rv_messedup=val_metrics_MCWts.compute_and_log(1,'val', 0, 3, False, None)
    rv_messedup2=val_metrics_MCWts2.compute_and_log(1,'val', 0, 3, False, None)



# %%
val_dataloader._reset_indices()
batch = next(val_dataloader)
x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types).squeeze()
n=10
print((outputs.argmax(dim=-1) * (types!=padding_token).to(int))[:n])
print((x[:n,:,-1]>1)*2 + (x[:n,:,-1]==1))
# print(x[:n,:,-1].to(int))
print(dsids[:n])
print(y.argmax(dim=-1)[:n])


# %%
# if 0:
#     import attachModelHooks
#     from importlib import reload # In case we make changes, for development only
#     reload(attachModelHooks)
#     mi = attachModelHooks.ModelInterpreter(model, device)
#     fig, attn_pattern = mi.get_attention_patterns(batch, num_samples=10)
#     # for ind in range(5):
#     #     print('----------------------------------------------------')
#     #     print(batch['x'][ind,:,-1].to(int))
#     #     print(batch['types'][ind].to(int))
#     #     print(outputs[ind].argmax(dim=-1).to(int))
#     # fig.show()
#     if SHUFFLE_OBJECTS:
#         fig.savefig('_Shuffled.png')
#     else:
#         fig.savefig('_NonShuffled.png')

#     print("Look at index 2 (event #3) here! Seems like it might be considering reconstructing the Higgs as a pair of small-R jets...")

#     # %%
#     fig, res = mi.analyze_residuals(batch, num_samples=20)

#     # %%
#     from importlib import reload # In case we make changes, for development only
#     reload(attachModelHooks)
#     mi = attachModelHooks.ModelInterpreter(model, device)
#     fig, corr_matrix = mi.feature_importance(batch)

#     # %%
#     fig, corr_matrix = mi.truth_label_impact(batch)

#     # %%
#     fig, corr_matrix = mi.ablation_study(batch)



#     # %%
#     import runMechInterp





#     # %%
#     ind=0
#     print(batch['x'][ind,:,-1].to(int))
#     print(batch['types'][ind].to(int))
#     print(outputs[ind].argmax(dim=-1).to(int))
#     # tensor([3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     # tensor([2, 1, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5])
#     # tensor([2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2])

#     ind=1
#     print(batch['x'][ind,:,-1].to(int))
#     print(batch['types'][ind].to(int))
#     print(outputs[ind].argmax(dim=-1).to(int))
#     # tensor([3, 3, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
#     # tensor([2, 0, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5])
#     # tensor([2, 2, 0, 0, 1, 0, 0, 1, 0, 2, 2, 2])

#     # %%
#     ind=9
#     print(ind)
#     print(outputs[ind].argmax(dim=-1).to(int))
#     print(torch.softmax(outputs, dim=-1)[ind, torch.arange(len(outputs[ind])), outputs[ind].argmax(dim=-1).to(int)])
#     print((torch.softmax(outputs, dim=-1)[ind, torch.arange(len(outputs[ind])), outputs[ind].argmax(dim=-1).to(int)]*100).to(int))
#     # print(torch.softmax(outputs, dim=-1)[ind])

#     # %% 
#     from utils import Get_PtEtaPhiM_fromXYZT
#     ind = 2
#     ljh = x[ind,2, :4]
#     sjh1= x[ind,3, :4]
#     sjh2= x[ind,9, :4]
#     sjch = sjh1 + sjh2
#     ljhli = Get_PtEtaPhiM_fromXYZT(ljh[0], ljh[1], ljh[2], ljh[3], use_torch=True)
#     sjchli = Get_PtEtaPhiM_fromXYZT(sjch[0], sjch[1], sjch[2], sjch[3], use_torch=True)
#     sjh1li = Get_PtEtaPhiM_fromXYZT(sjh1[0], sjh1[1], sjh1[2], sjh1[3], use_torch=True)
#     sjh2li = Get_PtEtaPhiM_fromXYZT(sjh2[0], sjh2[1], sjh2[2], sjh2[3], use_torch=True)
#     print(ljhli)
#     print(sjchli)
#     print(sjh1li)
#     print(sjh2li)

# %%
importlib.reload(mechinterputils)

COMBINE_LEPTONS_FOR_PLOTS=False
with torch.no_grad():
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
    residuals = mechinterputils.get_residual_stream(cache)

    dla=mechinterputils.old_direct_logit_attribution(model, cache)
    ndla=mechinterputils.direct_logit_attribution(model, cache)
    apa=mechinterputils.analyze_attention_patterns(cache,0)
    ota=mechinterputils.analyze_object_type_attention(model, cache, types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS)
    ota_selfex=mechinterputils.analyze_object_type_attention(model, cache, types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS, exclude_self=True)

# %%
from IPython.display import display
import circuitsvis as cv
from utils import Get_PtEtaPhiM_fromXYZT, print_vars, print_inclusion
chosen_sample_index = 2
val_dataloader._reset_indices()
batch = next(val_dataloader)
x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
# cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
# outputs, cache = mechinterputils.run_with_cache_and_singleAttention(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
for chosen_sample_index in range(10,20):
    print("-------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")
    SHOW_ALL=False
    outs = model(x[chosen_sample_index:chosen_sample_index+1,:, :N_Real_Vars-int(EXCLUDE_TAG)], types[chosen_sample_index:chosen_sample_index+1,:]) 
    print(f"True inclusion: {print_inclusion(x[chosen_sample_index:chosen_sample_index+1,:, -1].squeeze(), types[chosen_sample_index:chosen_sample_index+1,:].squeeze())}")
    print(f"Pred inclusion: {print_inclusion(outs.argmax(dim=-1).squeeze(), types[chosen_sample_index:chosen_sample_index+1,:].squeeze())}")
    reconstructed_object_types = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'large-R jet', 4: 'small-R jet', 5:'None'}
    for layer in range(len(model.attention_blocks)):
        # attention_pattern = cache[f'block_{layer}_attention']['attn_weights'][chosen_sample_index]
        attention_pattern = cache[f'block_{layer}_attention']['attn_weights_per_head'][chosen_sample_index]
        objselection = (types[chosen_sample_index:chosen_sample_index+1]!=(N_CTX-1)).flatten() # This allows us to ignore the 'pad' tokens (ie the 'nothing' particles)
        if SHOW_ALL:
            my_names=[f"{reconstructed_object_types[types[chosen_sample_index, obj].item()]:10} (E={x[chosen_sample_index,obj,3]:.2f})(M={Get_PtEtaPhiM_fromXYZT(x[chosen_sample_index,obj,0], x[chosen_sample_index,obj,1], x[chosen_sample_index,obj,2], x[chosen_sample_index,obj,3], use_torch=True)[-1]:.2f})" for obj in range(attention_pattern.shape[2])]
        else:
            attention_pattern = attention_pattern[:,objselection][:,:,objselection]
            my_names=[f"{reconstructed_object_types[types[chosen_sample_index, objselection][obj].item()]:10} (E={x[chosen_sample_index,objselection,3][obj]:.2f})(M={Get_PtEtaPhiM_fromXYZT(x[chosen_sample_index,objselection,0][obj], x[chosen_sample_index,objselection,1][obj], x[chosen_sample_index,objselection,2][obj], x[chosen_sample_index,objselection,3][obj], use_torch=True)[-1]:.2f})" for obj in range(attention_pattern.shape[2])]
        
        val_dataloader._reset_indices()
        display(
            cv.attention.attention_patterns(
                tokens=my_names,  # type: ignore
                attention=attention_pattern,
                # attention_head_names=[f"L0H{i}" for i in range(model.cfg.n_heads)],
            )
        )

# %%
if 1: # See how many of the attention weights are far from 1.0 or 0.0
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[0].self_attention.num_heads):
            for sample_idx in range(10):
                print(f"{cache[f'block_{layer}_attention']['attn_weights_per_head'][sample_idx, head][cache[f'block_{layer}_attention']['attn_weights_per_head'][sample_idx,head]<0.9].max().item():.2e}", end="   ")
            print()

# %%
# DO THIS AGAIN USING CIRCUITSVIS
if COMBINE_LEPTONS_FOR_PLOTS:
    types_dict = {0: 'lepton', 1: 'neutrino', 2: 'large-R jet', 3: 'small-R jet', 4:'None'}
else:
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'large-R jet', 4: 'small-R jet', 5:'None'}
num_layers = model.num_attention_blocks
num_heads = model.attention_blocks[0].self_attention.num_heads
if 0:
    for layer in range(num_layers):
        plt.figure(figsize=(6,3))
        for head in range(num_heads):
            plt.subplot(1,num_heads,head+1)
            plt.imshow(ota[f"block_{layer}"][f"head_{head}"])
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks(list(types_dict.keys()), list(types_dict.values()), rotation=90, size=10)
            plt.yticks(list(types_dict.keys()), list(types_dict.values()), rotation=0, size=10)
        plt.suptitle(f"Layer: {layer}")
        plt.tight_layout()
        plt.show()
else:
    for selfex in [False, True]:
        plt.figure(figsize=(3*num_heads,3*num_layers))
        for layer in range(num_layers):
            for head in range(num_heads):
                plt.subplot(num_layers,num_heads,layer*num_heads+head+1)
                if selfex:
                    plot_data = ota_selfex[f"block_{layer}"][f"head_{head}"]
                else:
                    plot_data = ota[f"block_{layer}"][f"head_{head}"]
                plt.imshow(plot_data)
                for (j,i),label in np.ndenumerate(plot_data):
                    plt.text(i,j,f"{int(label*100):2d}",ha='center',va='center',size=10)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.xticks(list(types_dict.keys()), list(types_dict.values()), rotation=90, size=10)
                plt.yticks(list(types_dict.keys()), list(types_dict.values()), rotation=0, size=10)
                plt.title(f"Layer {layer}, Head {head}")
        plt.suptitle("Average type attention patterns (rows are queries, cols are keys)")
        plt.tight_layout()
        plt.savefig(f'tmp8{selfex*"_selfex"}.pdf')
        plt.show()


# %%
if 0: # Cell to show the attention output sizes (Loop over layer/head, absolute then sum along the dmodel dimension, mean along the objects (excluding Nones) and then batch)
    #   Also has per-object
    import einops

    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
    
    # First, the object net (direct from embedding) sizes...
    absolute_activations_bo = cache['object_net']['output'].abs().mean(dim=-1)
    print(f"Inp: {(einops.einsum(absolute_activations_bo * (types!=(N_CTX-1)), 'batch object -> batch') / einops.einsum(types!=(N_CTX-1), 'batch object -> batch')).mean()}")
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[0].self_attention.num_heads):
            # print(f"{layer}.{head}: {cache[f'block_{layer}_attention']['attn_output_per_head'][:,head,...].abs().mean(dim=-1)[...,0].mean()}") # Just the neutrinos
            absolute_activations_bo = cache[f'block_{layer}_attention']['attn_output_per_head'][:,head,...].abs().mean(dim=-1) # Shape [batch object]
            print(f"{layer}.{head}: {(einops.einsum(absolute_activations_bo * (types!=(N_CTX-1)), 'batch object -> batch') / einops.einsum(types!=(N_CTX-1), 'batch object -> batch')).mean()}")
        if model.include_mlp:
            absolute_activations_bo = cache[f'block_{layer}_post_attention']['output'].abs().mean(dim=-1) # Shape [batch object]
            print(f"{layer}mlp {(einops.einsum(absolute_activations_bo * (types!=(N_CTX-1)), 'batch object -> batch') / einops.einsum(types!=(N_CTX-1), 'batch object -> batch')).mean()}")
    # And per object:
    print("         ", end="")
    for object_type in types_dict.keys():
        print(f"{types_dict[object_type]:15s}", end="")
    print()

    # First, the object net (direct from embedding) sizes...
    absolute_activations_bo = cache['object_net']['output'].abs().mean(dim=-1)
    print(f"Inp: ", end='')
    for object_type in types_dict.keys():
        object_type_mask = types == object_type
        object_counts = einops.einsum(object_type_mask, 'batch object -> batch')
        object_counts[object_counts==0] = 1 # Avoid divide by 0; the numerator will be 0 anyway
        print(f"    {(einops.einsum(absolute_activations_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():8.4f}  ", end="")
    print()
    # Now the layers
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[0].self_attention.num_heads):
            print(f"{layer}.{head}: ", end='')
            absolute_activations_bo = cache[f'block_{layer}_attention']['attn_output_per_head'][:,head,...].abs().mean(dim=-1) # Shape [batch object]
            # for object_type in [0]:
            for object_type in types_dict.keys():
                object_type_mask = types == object_type
                object_counts = einops.einsum(object_type_mask, 'batch object -> batch')
                object_counts[object_counts==0] = 1 # Avoid divide by 0; the numerator will be 0 anyway
                # print(f"{types_dict[object_type]:10s}: {(einops.einsum(absolute_activations_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():6.4f}", end="")
                print(f"    {(einops.einsum(absolute_activations_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():8.4f}  ", end="")
            print()
        if model.include_mlp:
            print(f"{layer}mlp ", end='')
            absolute_activations_bo = cache[f'block_{layer}_post_attention']['output'].abs().mean(dim=-1) # Shape [batch object]
            # for object_type in [0]:
            for object_type in types_dict.keys():
                object_type_mask = types == object_type
                object_counts = einops.einsum(object_type_mask, 'batch object -> batch')
                object_counts[object_counts==0] = 1 # Avoid divide by 0; the numerator will be 0 anyway
                # print(f"{types_dict[object_type]:10s}: {(einops.einsum(absolute_activations_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():6.4f}", end="")
                print(f"    {(einops.einsum(absolute_activations_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():8.4f}  ", end="")
            print()

# %%
if 1: 
    # Cell to show the absolute magnitude of the contributions to the direction of each of the possible inclusions,
    # at different points in the model
    # (Loop over layer/head, absolute then sum along the dmodel dimension, mean along the objects (excluding Nones) and then batch)
    #   Also has per-object
    import einops

    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
    
    for direction in range(model.classifier[0].weight.shape[0]):
        print(f"Direction: {direction}")
        dirn = model.classifier[0].weight[direction]
        # First, the object net (direct from embedding) sizes...
        absolute_logit_contributions_bo = einops.einsum(cache['object_net']['output'], dirn, 'batch object dmodel, dmodel -> batch object').abs()
        print(f"Inp: {(einops.einsum(absolute_logit_contributions_bo * (types!=(N_CTX-1)), 'batch object -> batch') / einops.einsum(types!=(N_CTX-1), 'batch object -> batch')).mean()}")
        for layer in range(len(model.attention_blocks)):
            for head in range(model.attention_blocks[0].self_attention.num_heads):
                # print(f"{layer}.{head}: {cache[f'block_{layer}_attention']['attn_output_per_head'][:,head,...].abs().mean(dim=-1)[...,0].mean()}") # Just the neutrinos
                absolute_logit_contributions_bo = einops.einsum(cache[f'block_{layer}_attention']['attn_output_per_head'][:,head,...], dirn, 'batch object dmodel, dmodel -> batch object').abs() # Shape [batch object]
                print(f"{layer}.{head}: {(einops.einsum(absolute_logit_contributions_bo * (types!=(N_CTX-1)), 'batch object -> batch') / einops.einsum(types!=(N_CTX-1), 'batch object -> batch')).mean()}")
            if model.include_mlp:
                absolute_logit_contributions_bo = einops.einsum(cache[f'block_{layer}_post_attention']['output'], dirn, 'batch object dmodel, dmodel -> batch object').abs() # Shape [batch object]
                print(f"{layer}mlp {(einops.einsum(absolute_logit_contributions_bo * (types!=(N_CTX-1)), 'batch object -> batch') / einops.einsum(types!=(N_CTX-1), 'batch object -> batch')).mean()}")

        # And per object:
        print("         ", end="")
        for object_type in types_dict.keys():
            print(f"{types_dict[object_type]:15s}", end="")
        print()

        # First, the object net (direct from embedding) sizes...
        absolute_logit_contributions_bo = einops.einsum(cache['object_net']['output'], dirn, 'batch object dmodel, dmodel -> batch object').abs()
        print(f"Inp: ", end='')
        for object_type in types_dict.keys():
            object_type_mask = types == object_type
            object_counts = einops.einsum(object_type_mask, 'batch object -> batch')
            object_counts[object_counts==0] = 1 # Avoid divide by 0; the numerator will be 0 anyway
            print(f"    {(einops.einsum(absolute_logit_contributions_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():8.4f}  ", end="")
        print()
        # Now the layers
        for layer in range(len(model.attention_blocks)):
            for head in range(model.attention_blocks[0].self_attention.num_heads):
                print(f"{layer}.{head}: ", end='')
                absolute_logit_contributions_bo = einops.einsum(cache[f'block_{layer}_attention']['attn_output_per_head'][:,head,...], dirn, 'batch object dmodel, dmodel -> batch object').abs() # Shape [batch object]
                # for object_type in [0]:
                for object_type in types_dict.keys():
                    object_type_mask = types == object_type
                    object_counts = einops.einsum(object_type_mask, 'batch object -> batch')
                    object_counts[object_counts==0] = 1 # Avoid divide by 0; the numerator will be 0 anyway
                    # print(f"{types_dict[object_type]:10s}: {(einops.einsum(absolute_logit_contributions_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():6.4f}", end="")
                    print(f"    {(einops.einsum(absolute_logit_contributions_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():8.4f}  ", end="")
                print()
            if model.include_mlp:
                print(f"{layer}mlp:", end='')
                absolute_logit_contributions_bo = einops.einsum(cache[f'block_{layer}_post_attention']['output'], dirn, 'batch object dmodel, dmodel -> batch object').abs() # Shape [batch object]
                # for object_type in [0]:
                for object_type in types_dict.keys():
                    object_type_mask = types == object_type
                    object_counts = einops.einsum(object_type_mask, 'batch object -> batch')
                    object_counts[object_counts==0] = 1 # Avoid divide by 0; the numerator will be 0 anyway
                    # print(f"{types_dict[object_type]:10s}: {(einops.einsum(absolute_logit_contributions_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():6.4f}", end="")
                    print(f"    {(einops.einsum(absolute_logit_contributions_bo * object_type_mask, 'batch object -> batch') / object_counts).mean():8.4f}  ", end="")
                print()

        print('------------------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------------------------')

# %%
mechinterputils.current_attn_detector(model, cache)

# %%
if 1:
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)

    importlib.reload(mechinterputils)
    for direction in range(3):
        print(f"Direction: {direction}")
        i=mechinterputils.likelyCandidate_attn_detector(model, cache, direction, padding_token=N_CTX-1)
        print("")
        print("")


    # And now same thing but restrict to specific objects
    for direction in range(3):
        print(f"Direction: {direction}")
        for object_type in range(5):
            print(types_dict[object_type])
            i=mechinterputils.likelyCandidate_attn_detector(model, cache, direction, padding_token=N_CTX-1, object_types_to_include=[object_type])
            print("")
        print("")
        print("")


# %%
importlib.reload(mechinterputils)
# mechinterputils.angular_separation_detector(model, cache, x, types, padding_token)
if 0:
    ah, scores = mechinterputils.angular_separation_detector_split_by_type(model, cache, x, types, padding_token, layers=[2], heads=[1], query_types=[0,1,2], key_types=[3])
elif 0:
    ah, scores = mechinterputils.angular_separation_detector_split_by_type(model, cache, x, types, padding_token, layers=[2], heads=[1], query_types=[2], key_types=[0, 1])
elif 0:
    ah, scores = mechinterputils.angular_separation_detector_split_by_type(model, cache, x, types, padding_token, layers=None, heads=None, query_types=[2], key_types=[0, 1])

# %%
print("Really should add code to do the same for background samples here/incldue them in some calls, since they have truth_inclusion all set to zero so this skews the results")
print("In reality, should probably move the cachce-getting intot he plot_logit_attributions function and have it loop until it gets a sufficient number of samples (to be specified)")
importlib.reload(mechinterputils)
with torch.no_grad():
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, _, types, _, _, _, _, _ = batch.values()
    x = x[y.argmax(dim=-1)>0]
    types = types[y.argmax(dim=-1)>0]
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
for obj_type in [0,1,2]:
    # _=mechinterputils.plot_logit_attributions(model, cache, x[...,-1], types, [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2)", include_mlp=HAS_MLP, include_direct_from_embedding=True)
    _=mechinterputils.plot_logit_attributions(model, cache, x[...,-1], types, [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2) [NO MLP]", include_mlp=False, include_direct_from_embedding=True)
    _=mechinterputils.plot_logit_attributions(model, cache, x[...,-1], types, [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2) [WITH MLP]", include_mlp=True, include_direct_from_embedding=True)

# %%

for obj_type in [0,1,2]:
    _=mechinterputils.plot_logit_attributionsMultiBatch(model, val_dataloader, N_Real_Vars-int(EXCLUDE_TAG), [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2)", include_mlp=HAS_MLP, include_direct_from_embedding=True, min_samples=100)#, dsid_incl=None)

# %%
with torch.no_grad():
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, _, types, _, _, _, _, _ = batch.values()
    x = x[y.argmax(dim=-1)>0]
    types = types[y.argmax(dim=-1)>0]
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
for obj_type in [0,1,2]:
    _=mechinterputils.plot_logit_attributions(model, cache, x[...,-1], types, [0], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to neither particle H+ decay product (class 0)", include_mlp=HAS_MLP)

# %%
with torch.no_grad():
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, _, types, _, _, _, _, _ = batch.values()
    x = x[y.argmax(dim=-1)>0]
    types = types[y.argmax(dim=-1)>0]
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
truth_inclusion_mapping = {
    0:"neither particle",
    1:"SM Higgs",
    2:"W boson",
}
for true_incl in range(3):
    _=mechinterputils.plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [3], title=f"Logit attribution for predicting reco large-R jets, truth-matched\nto be part of {truth_inclusion_mapping[true_incl]} (class {true_incl}) from H+", include_mlp=HAS_MLP)


# %%
with torch.no_grad():
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, _, types, _, _, _, _, _ = batch.values()
    x = x[y.argmax(dim=-1)>0]
    types = types[y.argmax(dim=-1)>0]
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
for true_incl in range(3):
    _=mechinterputils.plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [4], title=f"Logit attribution for predicting reco small-R jets, truth-matched\nto be part of {truth_inclusion_mapping[true_incl]} (class {true_incl}) from H+", include_mlp=HAS_MLP)



# %%
# Observe the bits which tell a electron/muon/neutrino which comes from a W+ in a signal event where it comes from, for highest vs. lowest signal masses. Broadly the same pattern, but notice more info that it's NOT in the SM higgs comes from layer3-attentionhead0 in the higher mass case (though not usper interesting since it can never come from SM higgs in true signal, so it's never seen this in training).
for obj_type in [0,1,2]:
    _=mechinterputils.plot_logit_attributionsMultiBatch(model, val_dataloader, N_Real_Vars-int(EXCLUDE_TAG), [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2), dsid=510115", include_mlp=HAS_MLP, include_direct_from_embedding=True, dsid_incl=[510115], min_samples=100)
    _=mechinterputils.plot_logit_attributionsMultiBatch(model, val_dataloader, N_Real_Vars-int(EXCLUDE_TAG), [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2), dsid=510124", include_mlp=HAS_MLP, include_direct_from_embedding=True, dsid_incl=[510124], min_samples=100)
    
# %%
# Similar but now we observe the contributions from different components to the logits for class 0/1/2 in a electron/muon/neutrino which comes from nothing, for background ttbar sample
for obj_type in [0,1,2]:
    _=mechinterputils.plot_logit_attributionsMultiBatch(model, val_dataloader, N_Real_Vars-int(EXCLUDE_TAG), [0], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, ttbar events (so no truth matching)", include_mlp=HAS_MLP, include_direct_from_embedding=True, dsid_incl=[410470], min_samples=100)
# %%
# %%
# %%
# %%
# %%
# %%

with torch.no_grad():
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, _, types, _, _, _, _, _ = batch.values()
    cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
val_dataloader.N_Real_Vars_To_Return
model.object_net
# %%
if 1: # Test what happens if we add random objects...
    from utils import print_inclusion
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, _, types, _, _, _, _, _ = batch.values()
    orig_outputs = model(x[..., :N_Real_Vars-int(EXCLUDE_TAG)], types)
    # Make new inputs with the last small-R jet replaced with lepton
    is_sjet = (types==4).to(int)
    last_sjet_index = (is_sjet.size(1) - 1) - torch.argmax(is_sjet.flip(dims=[1]), dim=1)
    last_sjet_index
    types_new = types.clone()
    types_new[torch.arange(types.shape[0]),last_sjet_index] = 2
    new_outputs = model(x[..., :N_Real_Vars-int(EXCLUDE_TAG)], types_new)

    for batch_idx in range(10):
        print(f"True inclusion:                     {print_inclusion(x[batch_idx:batch_idx+1,:, -1].squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
        print(f"Pred inclusion before intervention: {print_inclusion(orig_outputs[batch_idx:batch_idx+1].argmax(dim=-1).squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
        # print(f"Pred inclusion after  intervention: {print_inclusion(new_outputs[batch_idx:batch_idx+1].argmax(dim=-1).squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
        print(f"Pred inclusion after  intervention: {print_inclusion(new_outputs[batch_idx:batch_idx+1].argmax(dim=-1).squeeze(), types_new[batch_idx:batch_idx+1,:].squeeze())}")
        print('-----------------------------------------------------------------------------------------------------')
    
    


    # cache = mechinterputils.extract_all_activations(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)

# %%
if 0: # Takes a while as it runs a lot of symbolic regression
    importlib.reload(mechinterputils)
    results = mechinterputils.main_sae_analysis(model, val_dataloader, N_Real_Vars, device=device, hidden_dim_sae=2)

# %%
if 1: # Takes a while as it runs a lot of symbolic regression
    importlib.reload(mechinterputils)
    results = mechinterputils.main_symbolic_regression(model, val_dataloader, N_Real_Vars-int(EXCLUDE_TAG), device=device)

# %%
inverse_types_dict = {types_dict[type_key]:type_key for type_key in types_dict.keys()}
block_num=0
if 1: # Not using yet as currently can only look for eg. Pt, Eta, which we are already giving as inputs!
    importlib.reload(mechinterputils)
    for block_num in range(3):
        # for head in range(4):
        head=None
        metrics = mechinterputils.learned_feature_probe(model, 
                            val_dataloader, 
                            N_Real_Vars, 
                            [2],
                            # [0,1],
                            # [3],
                            object_type_mask_for_query=True,
                            # probe_target='delta-eta-with-lepton',
                            # probe_target='delta-phi-with-lepton',
                            # probe_target='delta-R-with-lepton',
                            # probe_target='delta-eta-with-neutrino',
                            # probe_target='delta-phi-with-neutrino',
                            # probe_target='delta-R-with-neutrino',
                            # probe_target='delta-eta-with-largeRjet',
                            # probe_target='delta-phi-with-largeRjet',
                            probe_target='delta-R-with-largeRjet',
                            # probe_target='delta-Rs-with-largeRjet',
                            #    probe_layer='object_net',
                            # probe_layer=f'block_{block_num}_attention',
                            probe_layer=f'residuals_{block_num}',
                            head=head,
                            verbose=3,
                            num_epochs=20000,
                            learning_rate=1e-3,
                            restrict_to_single_ljet_events=True,
                            )

# %%
if 0: # Doesn't currently seem to work & doesn't seem to plan to do much more than visualise the attention pattern
    importlib.reload(mechinterputils)
    flow_metrics = mechinterputils.track_information_flow(model, val_dataloader, N_Real_Vars)
    mechinterputils.visualize_information_flow(flow_metrics, save_path='information_flow.png')
    # flow_metrics = track_information_flow(model, data_loader, n_inputs)
    # visualize_information_flow(flow_metrics, save_path='information_flow.png')

# %%
if 0: # Not tested yet, working out if the activation_maximization_for_attention_heads_KeepFixed version is better
    importlib.reload(mechinterputils)
    batch_idx = len(x)-1
    mechinterputils.activation_maximization_for_attention_heads(model, x[batch_idx:batch_idx+1,:, :N_Real_Vars-int(EXCLUDE_TAG)], types[batch_idx:batch_idx+1,...], 0, 2, target_head=1)




# %%
from utils import print_vars, print_inclusion
if 1:
    # Start of the batch will be signal, end of the batch will be bkg
    importlib.reload(mechinterputils)
    batch_idx = len(x)-1
    batch_idx = 3
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    # initial_in = x[batch_idx:batch_idx+1,:, :N_Real_Vars-int(EXCLUDE_TAG)]
    initial_in = x[batch_idx:batch_idx+1,:, :N_Real_Vars]
    outs = model(x[batch_idx:batch_idx+1,:, :N_Real_Vars], types[batch_idx:batch_idx+1,:])
    if 1: # Look from the neutrino, at the electron/muon, and maximise the chance that we identify the neutrino as being part of the W-boson
        object_index_to_maximise = 1
        object_index_to_change = 0
        # object_direction_to_change = (0,2)
        object_direction_to_change = (2,0)
        target_layer = 0
        target_head = 3 # Head, or 'None' for all heads aggregated
        lr=3e-3
        iters=1000
        # iters=500
        # iters=50
    maxed_in, scores, altered_scores, unaltered_scores = mechinterputils.activation_maximization_for_attention_heads_KeepFixed(model, initial_in, types[batch_idx:batch_idx+1,...], object_index_to_maximise, object_index_to_change, target_layer, target_head=target_head, maximise_direction=object_direction_to_change, lr=lr, iters=iters, print_every=100)
    # maxed_in=mechinterputils.activation_maximization_for_attention_heads_KeepFixed(model, initial_in, types[:1,...], 1, 0, target_head=1, maximise_direction=2)
    # maxed_in=mechinterputils.activation_maximization_for_attention_heads_KeepFixed(model, initial_in, types[:1,...], 0, 0, target_head=1,lr=3e-3,iters=2000)
    # print(initial_in)
    # print(maxed_in)
    # new_vars = torch.stack((*Get_PtEtaPhiM_fromXYZT(maxed_in[...,0],maxed_in[...,1],maxed_in[...,2],maxed_in[...,3],use_torch=True), maxed_in[...,4]),dim=-1)
    plt.close('all')
    plt.figure(figsize=(3,2))
    plt.plot(scores)
    plt.show()

    print(f"True inclusion: {print_inclusion(x[batch_idx:batch_idx+1,:, -1].squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
    print(f"Pred inclusion: {print_inclusion(outs.argmax(dim=-1).squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
    print_vars(initial_in, types[batch_idx:batch_idx+1,...])
    print('------------------------------------------------------------')
    # print_vars(maxed_in, types[:1,...])
    print_vars(maxed_in[:,object_index_to_change:object_index_to_change+1,:], types[batch_idx:batch_idx+1,object_index_to_change:object_index_to_change+1])
    if 1:
        print(altered_scores)
        print(unaltered_scores)
    if 0: # Good for if we're looking at lepton/neutrino interactions
        # PRINT THE sum of the two things (pt, eta, phi, m) to see if there's any pattern
        print("Original Wlv: ", end='')
        print_vars(initial_in[:,object_index_to_change:object_index_to_change+1,:]+initial_in[:,object_index_to_maximise:object_index_to_maximise+1,:], types[batch_idx:batch_idx+1,object_index_to_change:object_index_to_change+1])
        print("Maxed Wlv: ", end='')
        print_vars(maxed_in[:,object_index_to_change:object_index_to_change+1,:]+maxed_in[:,object_index_to_maximise:object_index_to_maximise+1,:], types[batch_idx:batch_idx+1,object_index_to_change:object_index_to_change+1])
    # print()
    # print('------------------------------------------------------------')
    # print()


# %%

# %%
from utils import print_vars, print_inclusion
for batch_idx in range(10):
    outs = model(x[batch_idx:batch_idx+1,:, :N_Real_Vars-int(EXCLUDE_TAG)], types[batch_idx:batch_idx+1,:]).argmax(dim=-1)
    # print(outs, y[batch_idx].argmax(dim=-1))
    channel_mapping = {0:'None', 1:'lvbb', 2:'qqbb'}
    print(f"Truth: {channel_mapping[y[batch_idx].argmax(dim=-1).item()]}")
    print(f"True inclusion: {print_inclusion(x[batch_idx:batch_idx+1,:, -1].squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
    print(f"Pred inclusion: {print_inclusion(outs.squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')
for batch_idx in range(-10,-1):
    outs = model(x[batch_idx:batch_idx+1,:, :N_Real_Vars-int(EXCLUDE_TAG)], types[batch_idx:batch_idx+1,:]).argmax(dim=-1)
    # print(outs, y[batch_idx].argmax(dim=-1))
    print(f"Truth: {channel_mapping[y[batch_idx].argmax(dim=-1).item()]}")
    print(f"True inclusion: {print_inclusion(x[batch_idx:batch_idx+1,:, -1].squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
    print(f"Pred inclusion: {print_inclusion(outs.squeeze(), types[batch_idx:batch_idx+1,:].squeeze())}")
    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')




# %%
if 0: # Now an ablation study
    importlib.reload(mechinterputils)
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    layer=0
    head=0
    interventions = [
        (f'attention_blocks.{layer}.self_attention', lambda x: mechinterputils.ablate_attention_head(x, head_idx=0, num_heads=model.attention_blocks[0].self_attention.num_heads))
    ]

    # Run the model with interventions
    orig_output, orig_cache = mechinterputils.run_with_interventions(model, (x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types), [])
    ablated_output, ablated_cache = mechinterputils.run_with_interventions(model, (x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types), interventions)
    batch_idx = 0
    print(orig_output[batch_idx, types[batch_idx]!=(N_CTX-1)])
    print(ablated_output[batch_idx, types[batch_idx]!=(N_CTX-1)])
    print(orig_output[batch_idx, types[batch_idx]!=(N_CTX-1)] - ablated_output[batch_idx, types[batch_idx]!=(N_CTX-1)])



# %%
# Full ablation study
# Cell to run over attention heads and ablate to see model performance
# Define a helper 'evaluate' function
def model_eval(model, dataloader, interventions, verbose=False):
    metric_tracker = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(dataloader.get_total_samples()), total_weights_per_dsid=dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
    dataloader._reset_indices()
    metric_tracker.reset()
    model.eval()
    # num_samples = 30000
    num_samples = 5000
    num_batches_to_process = int(num_samples * (1/batch_size))
    for batch_idx in range(num_batches_to_process):
        if ((batch_idx%10)==9) and verbose:
            print(F"Processing batch {batch_idx}/{num_batches_to_process}")
        batch = next(dataloader)
        x, _, _, types, dsids, _, _, MCWts, _ = batch.values()
        outputs, _ = mechinterputils.run_with_interventions(model, (x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types), interventions=interventions)
        metric_tracker.update(outputs, x[...,-1], MCWts, dsids, types)

    return metric_tracker.compute_and_log(1,'val', 0, 3, False, None, False, calc_all=True)

if 1: # Evaluate the model with different attention heads ablated
    results_single_ablation = {}
    results_single_ablation[tuple()]=rv
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[0].self_attention.num_heads):
            interventions = [
                (f'attention_blocks.{layer}.self_attention', lambda x: mechinterputils.ablate_attention_head(x, head_idx=head, num_heads=model.attention_blocks[0].self_attention.num_heads))
            ]
            results_single_ablation[(layer, head)] = model_eval(model, val_dataloader, interventions)
            print(f"{layer}, {head}: lvbball={results_single_ablation[(layer, head)]['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_single_ablation[(layer, head)]['val/PerfectRecoPct_all_qqbb']:.4f}")
    if 0:
        for k in results_single_ablation.keys():
            print(f"{str(k):6s}: lvbball={results_single_ablation[k]['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_single_ablation[k]['val/PerfectRecoPct_all_qqbb']:.4f}")
    elif 0: # Print detailed results per channel
        print("              lvbb   Corr / Over / Undr / Miss\tqqbb   Corr / Over / Undr / Miss")
        for k in results_single_ablation.keys():
            print(f"{str(k):12s}: ", end="")
            print(f"lvbb = {results_single_ablation[k]['val/PerfectRecoPct_all_lvbb']*100:4.1f}", end="")
            print(f" / {results_single_ablation[k]['val/Pct_OverPredict_all_lvbb']*100:4.1f}", end="")
            print(f" / {results_single_ablation[k]['val/Pct_UnderPredict_all_lvbb']*100:4.1f}", end="")
            print(f" / {results_single_ablation[k]['val/Pct_MisPredict_all_lvbb']*100:4.1f}", end="\t")
            print(f"qqbb = {results_single_ablation[k]['val/PerfectRecoPct_all_qqbb']*100:4.1f}", end="")
            print(f" / {results_single_ablation[k]['val/Pct_OverPredict_all_qqbb']*100:4.1f}", end="")
            print(f" / {results_single_ablation[k]['val/Pct_UnderPredict_all_qqbb']*100:4.1f}", end="")
            print(f" / {results_single_ablation[k]['val/Pct_MisPredict_all_qqbb']*100:4.1f}")
    elif 1: # Print detailed results per reconstruction category
        # There are 6 ways to reconstruct an event correctly
        # 0: small-R-pair H, small-R-pair W
        # 1: small-R-pair H, leptonic-W
        # 2: small-R-pair H, large-R-pair W
        # 3: large-R-pair H, small-R-pair W
        # 4: large-R-pair H, leptonic-W
        # 5: large-R-pair H, large-R-pair W
        if 0: # All in one line
            for category in range(6):
                print(f'cat{category}   Corr / Over / Undr / Miss', end="\t")
            print()
            for k in results_single_ablation.keys():
                print(f"{str(k):12s}: ", end="")
                for category in range(6):
                    print(f"cat{category} = {results_single_ablation[k][f'val/PerfectRecoPct_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_OverPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_UnderPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_MisPredict_all_cat{category}']*100:4.1f}", end="\t")
                print()
        else: # Print in separate lines
            print(f'catego Corr / Over / Undr / Miss', end="\t")
            for k in results_single_ablation.keys():
                print(f"{str(k):12s}: ")
                for category in range(6):
                    print(f"cat{category} = {results_single_ablation[k][f'val/PerfectRecoPct_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_OverPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_UnderPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_MisPredict_all_cat{category}']*100:4.1f}")
                print()


# %%
if 1: # Evaluate the model with different PAIRS OF attention heads ablated
    results_pairs = {}
    results_pairs['no ablation']=rv
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[0].self_attention.num_heads):
            for layer2 in range(layer, len(model.attention_blocks)):
                for head2 in range(int(layer2==layer)*(head+1), model.attention_blocks[0].self_attention.num_heads):
                    interventions = [
                        (f'attention_blocks.{layer}.self_attention', lambda x: mechinterputils.ablate_attention_head(x, head_idx=head, num_heads=model.attention_blocks[0].self_attention.num_heads)),
                        (f'attention_blocks.{layer2}.self_attention', lambda x: mechinterputils.ablate_attention_head(x, head_idx=head2, num_heads=model.attention_blocks[0].self_attention.num_heads)),
                    ]
                    results_pairs[f'{layer},{head}  {layer2},{head2}'] = model_eval(model, val_dataloader, interventions)
                    print(f"{layer},{head},  {layer2},{head2}: lvbball={results_pairs[f'{layer},{head}  {layer2},{head2}']['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_pairs[f'{layer},{head}  {layer2},{head2}']['val/PerfectRecoPct_all_qqbb']:.4f}")
    if 0:
        for k in results_pairs.keys():
            print(f"{str(k):6s}: lvbball={results_pairs[k]['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_pairs[k]['val/PerfectRecoPct_all_qqbb']:.4f}")
    elif 0:
        print("              lvbb   Corr / Over / Undr / Miss\tqqbb   Corr / Over / Undr / Miss")
        for k in results_pairs.keys():
            print(f"{str(k):12s}: ", end="")
            print(f"lvbb = {results_pairs[k]['val/PerfectRecoPct_all_lvbb']*100:4.1f}", end="")
            print(f" / {results_pairs[k]['val/Pct_OverPredict_all_lvbb']*100:4.1f}", end="")
            print(f" / {results_pairs[k]['val/Pct_UnderPredict_all_lvbb']*100:4.1f}", end="")
            print(f" / {results_pairs[k]['val/Pct_MisPredict_all_lvbb']*100:4.1f}", end="\t")
            print(f"qqbb = {results_pairs[k]['val/PerfectRecoPct_all_qqbb']*100:4.1f}", end="")
            print(f" / {results_pairs[k]['val/Pct_OverPredict_all_qqbb']*100:4.1f}", end="")
            print(f" / {results_pairs[k]['val/Pct_UnderPredict_all_qqbb']*100:4.1f}", end="")
            print(f" / {results_pairs[k]['val/Pct_MisPredict_all_qqbb']*100:4.1f}")
            # print('-'*(12+2+len('lvbb   Corr / Over / Undr / Miss')+2+len('qqbb   Corr / Over / Undr / Miss')))
    elif 1: # Print detailed results per reconstruction category
        # There are 6 ways to reconstruct an event correctly
        # 0: small-R-pair H, small-R-pair W
        # 1: small-R-pair H, leptonic-W
        # 2: small-R-pair H, large-R-pair W
        # 3: large-R-pair H, small-R-pair W
        # 4: large-R-pair H, leptonic-W
        # 5: large-R-pair H, large-R-pair W
        if 0: # All in one line
            for category in range(6):
                print(f'cat{category}   Corr / Over / Undr / Miss', end="\t")
            print()
            for k in results_single_ablation.keys():
                print(f"{str(k):12s}: ", end="")
                for category in range(6):
                    print(f"cat{category} = {results_single_ablation[k][f'val/PerfectRecoPct_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_OverPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_UnderPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_MisPredict_all_cat{category}']*100:4.1f}", end="\t")
                print()
        else: # Print in separate lines
            print(f'catego Corr / Over / Undr / Miss', end="\t")
            for k in results_single_ablation.keys():
                print(f"{str(k):12s}: ")
                for category in range(6):
                    print(f"cat{category} = {results_single_ablation[k][f'val/PerfectRecoPct_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_OverPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_UnderPredict_all_cat{category}']*100:4.1f}", end="")
                    print(f" / {results_single_ablation[k][f'val/Pct_MisPredict_all_cat{category}']*100:4.1f}")
                print()


# %%
if 1: # Evaluate the model with different FULL LAYERS of attention heads removed
    results_layers = {}
    results_layers['no ablation']=rv
    for layer in range(len(model.attention_blocks)):
        interventions = [
            (f'attention_blocks.{layer}.self_attention', lambda x, h=h: mechinterputils.ablate_attention_head(x, head_idx=h, num_heads=model.attention_blocks[0].self_attention.num_heads)) for h in range(model.attention_blocks[0].self_attention.num_heads)
        ]
        results_layers[layer] = model_eval(model, val_dataloader, interventions)
        print(f"Layer {layer}: lvbball={results_layers[layer]['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_layers[layer]['val/PerfectRecoPct_all_qqbb']:.4f}")

for k in results_layers.keys():
    print(f"{str(k):6s}: lvbball={results_layers[k]['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_layers[k]['val/PerfectRecoPct_all_qqbb']:.4f}")

# %%
if 1: # Evaluate the model with ALL the attention heads removed
    results_all_removed = {}
    results_all_removed['no ablation']=rv
    interventions = []
    interventions+= [(f'attention_blocks.{0}.self_attention', lambda x, h=h: mechinterputils.ablate_attention_head(x, head_idx=h, num_heads=model.attention_blocks[0].self_attention.num_heads)) for h in range(model.attention_blocks[0].self_attention.num_heads)]
    interventions+= [(f'attention_blocks.{1}.self_attention', lambda x, h=h: mechinterputils.ablate_attention_head(x, head_idx=h, num_heads=model.attention_blocks[0].self_attention.num_heads)) for h in range(model.attention_blocks[0].self_attention.num_heads)]
    results_all_removed['All attn ablated'] = model_eval(model, val_dataloader, interventions)
    print(f"All ablated: lvbball={results_all_removed['All attn ablated']['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_all_removed['All attn ablated']['val/PerfectRecoPct_all_qqbb']:.4f}")

for k in results_all_removed.keys():
    print(f"{str(k):6s}: lvbball={results_all_removed[k]['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_all_removed[k]['val/PerfectRecoPct_all_qqbb']:.4f}")



# %%
# Cell to look at the Pt/Eta/Phi/M of some objects
from utils import Get_PtEtaPhiM_fromXYZT
n=1
# print(x[n][...,:3][x[n,:,-1]==1].sum(dim=0))
print('Higgs')
print([i.item() for i in Get_PtEtaPhiM_fromXYZT(*(x[n][...,i][x[n,:,-1]==1].sum(dim=0) for i in range(4)), use_torch=True)])
print('Higgs constituents:')
for j in range(x.shape[1]):
    if x[n,j,-1]==1:
        print([i.item() for i in Get_PtEtaPhiM_fromXYZT(*(x[n][j,i] for i in range(4)), use_torch=True)])

print('W boson')
# print(x[n][...,:3][x[n,:,-1]>1].sum(dim=0))
print([i.item() for i in Get_PtEtaPhiM_fromXYZT(*(x[n][...,i][x[n,:,-1]>1].sum(dim=0) for i in range(4)), use_torch=True)])
print('W boson constituents:')
for j in range(x.shape[1]):
    if x[n,j,-1]>1:
        print([i.item() for i in Get_PtEtaPhiM_fromXYZT(*(x[n][j,i] for i in range(4)), use_torch=True)])

# %%
# Cell to show the distribution of Pt/Eta/Phi/M of some objects
# Get all the values, then put in histograms with weights used to mask out
for type_key in list(types_dict.keys()) + ['all']:
    if type_key == 'all':
        type_mask = (types != N_CTX-1).to(int).flatten()
    else:
        type_mask = ((types == type_key) & (types != N_CTX-1)).to(int).flatten()
    is_in_higgs = (x[..., -1] == 1).to(int)
    is_in_W = (x[..., -1] > 1).to(int)
    is_in_neither = ((types!=N_CTX-1) & (x[..., -1] == 0)).to(int)
    pt, eta, phi, m = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
    Hpt, Heta, Hphi, Hm = Get_PtEtaPhiM_fromXYZT((x[...,0]* is_in_higgs).sum(dim=-1), (x[...,1]* is_in_higgs).sum(dim=-1), (x[...,2]* is_in_higgs).sum(dim=-1), (x[...,3]* is_in_higgs).sum(dim=-1), use_torch=True)
    Wpt, Weta, Wphi, Wm = Get_PtEtaPhiM_fromXYZT((x[...,0]* is_in_W).sum(dim=-1), (x[...,1]* is_in_W).sum(dim=-1), (x[...,2]* is_in_W).sum(dim=-1), (x[...,3]* is_in_W).sum(dim=-1), use_torch=True)
    pt, eta, phi, m, is_in_higgs, is_in_W, is_in_neither = pt.flatten(), eta.flatten(), phi.flatten(), m.flatten(), is_in_higgs.flatten(), is_in_W.flatten(), is_in_neither.flatten()
    plt.figure(figsize=(10,3))

    density=True

    plt.subplot(1,4,1)
    bins=np.linspace(0, 10, 100)
    plt.hist(pt, bins=bins, weights=is_in_higgs*type_mask, histtype='step', label='Higgs const.s', density=density)
    plt.hist(pt, bins=bins, weights=is_in_W*type_mask, histtype='step', label='W boson const.s', density=density)
    plt.hist(pt, bins=bins, weights=is_in_neither*type_mask, histtype='step', label='Other', density=density)
    plt.hist(Hpt, bins=bins, histtype='step', label='Higgs', density=density)
    plt.hist(Wpt, bins=bins, histtype='step', label='W', density=density)
    plt.title('Pt [100GeV]')
    plt.legend(prop={'size':5})

    plt.subplot(1,4,2)
    bins=np.linspace(-4,4,100)
    plt.hist(eta, bins=bins, weights=is_in_higgs*type_mask, histtype='step', label='Higgs const.s', density=density)
    plt.hist(eta, bins=bins, weights=is_in_W*type_mask, histtype='step', label='W boson const.s', density=density)
    plt.hist(eta, bins=bins, weights=is_in_neither*type_mask, histtype='step', label='Other', density=density)
    plt.hist(Heta, bins=bins, histtype='step', label='Higgs', density=density)
    plt.hist(Weta, bins=bins, histtype='step', label='W', density=density)
    plt.title('Eta')
    plt.legend(prop={'size':5})

    plt.subplot(1,4,3)
    bins=np.linspace(-3.14,3.14,100)
    plt.hist(phi, bins=bins, weights=is_in_higgs*type_mask, histtype='step', label='Higgs const.s', density=density)
    plt.hist(phi, bins=bins, weights=is_in_W*type_mask, histtype='step', label='W boson const.s', density=density)
    plt.hist(phi, bins=bins, weights=is_in_neither*type_mask, histtype='step', label='Other', density=density)
    plt.hist(Hphi, bins=bins, histtype='step', label='Higgs', density=density)
    plt.hist(Wphi, bins=bins, histtype='step', label='W', density=density)
    plt.title('Phi')
    plt.legend(prop={'size':5})

    plt.subplot(1,4,4)
    bins=np.linspace(0, 2, 100)
    plt.hist(m, bins=bins, weights=is_in_higgs*type_mask, histtype='step', label='Higgs const.s', density=density)
    plt.hist(m, bins=bins, weights=is_in_W*type_mask, histtype='step', label='W boson const.s', density=density)
    plt.hist(m, bins=bins, weights=is_in_neither*type_mask, histtype='step', label='Other', density=density)
    plt.hist(Hm, bins=bins, histtype='step', label='Higgs', density=density)
    plt.hist(Wm, bins=bins, histtype='step', label='W', density=density)
    plt.title('M [100GeV]')
    plt.legend(prop={'size':5})

    plt.tight_layout()
    if type_key == 'all':
        plt.suptitle(f"Type ALL ({type_key})")
    else:
        plt.suptitle(f"Type {types_dict[type_key]} ({type_key})")
    plt.show()

plt.close('all')


# %%
# Cell to see what happens if we change the angles between inputs
if 1:
    from utils import print_inclusion, print_vars#, GetXYZT_FromPtEtaPhiM
    model.eval()
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types).squeeze()
    # Find an event with the truth-W as two small-R jets
    is_sjet = (types==4).to(int)
    truth_W = x[..., -1] > 1
    a=(is_sjet & truth_W).sum(dim=-1)==2
    chosen_sample_index = torch.where(a)[0][2]
    
    print(f"True inclusion: {print_inclusion(x[chosen_sample_index:chosen_sample_index+1,:, -1].squeeze(), types[chosen_sample_index:chosen_sample_index+1,:].squeeze())}")
    print(f"Pred inclusion: {print_inclusion(outputs[chosen_sample_index].argmax(dim=-1).squeeze(), types[chosen_sample_index:chosen_sample_index+1,:].squeeze())}")

    print(outputs[chosen_sample_index, types[chosen_sample_index]!=N_CTX-1].transpose(0,1))

    # Now change the inputs by eg. making the second truth Higgs jet closer to the other one, and see what happens
    true_correct = (x[chosen_sample_index:chosen_sample_index+1,:, -1].squeeze() == outputs[chosen_sample_index].argmax(dim=-1).squeeze()) | (types[chosen_sample_index]==N_CTX-1)
    if 0:
        is_in_higgs = (x[chosen_sample_index, :, -1] == 1)
        assert((is_in_higgs & true_correct).sum(dim=-1)==1) # Have one out of two higgs small-R jets correct, otherwise this is not as interesting
        sjet1_correct = torch.where(is_in_higgs & true_correct)[0][0]
        sjet2_incorrect = torch.where(is_in_higgs & ~true_correct)[0][0]
    else:
        is_in_W = (x[chosen_sample_index, :, -1] > 1)
        assert((is_in_W & true_correct).sum(dim=-1)==1) # Have one out of two higgs small-R jets correct, otherwise this is not as interesting
        sjet1_correct = torch.where(is_in_W & true_correct)[0][0]
        sjet2_incorrect = torch.where(is_in_W & ~true_correct)[0][0]
    
    # print_vars(x[chosen_sample_index:chosen_sample_index+1,[sjet1_correct, sjet2_incorrect],:], types[chosen_sample_index:chosen_sample_index+1,[sjet1_correct, sjet2_incorrect]])
    print_vars(x[chosen_sample_index:chosen_sample_index+1,[3,4,5,6],:], types[chosen_sample_index:chosen_sample_index+1,[3,4,5,6]])

    Pt, Eta, Phi, M = Get_PtEtaPhiM_fromXYZT(*(x[chosen_sample_index,:,i] for i in range(4)), use_torch=True)

    # Change the angle between the two small-R jets
    # As a first naive attempt, just change the phi of the second small-R jet (incorrect one) to be the same as the first one (correctly identified)
    # print("Delta R before: ", torch.sqrt((Eta[])**2).sum(dim=-1).item())
    print(f"Phis before: {Phi[sjet1_correct].item():5.2f}, {Phi[sjet2_incorrect].item():5.2f}")
    Phi[sjet2_incorrect] = Phi[sjet1_correct]
    X, Y, Z, T = GetXYZT_FromPtEtaPhiM(Pt[sjet2_incorrect], Eta[sjet2_incorrect], Phi[sjet2_incorrect], M[sjet2_incorrect], use_torch=True)
    x[chosen_sample_index, sjet2_incorrect, :4] = torch.stack((X,Y,Z,T), dim=-1)
    
    # Now run the model again with this new input
    outputs_updated = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types).squeeze()
    print(f"True inclusion: {print_inclusion(x[
