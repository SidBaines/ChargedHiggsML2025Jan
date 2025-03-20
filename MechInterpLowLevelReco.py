# %% # Load required modules
import MechInterpUtils
import importlib 
import numpy as np
import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')
from mynewdataloader import ProportionalMemoryMappedDataset
from torch import nn
import matplotlib.pyplot as plt
import shutil
from MyMetricsLowLevelRecoTruthMatching import HEPMetrics

# %%
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")
# saveDir = "output/" + timeStr  + "_MechInterpOutput/"
# os.makedirs(saveDir)
# print(saveDir)


# Some choices about the training process
# Assumes that the data has already been binarised
IS_CATEGORICAL = True
PHI_ROTATED = True
REMOVE_WHERE_TRUTH_WOULD_BE_CUT = True
MODEL_ARCH="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP"
device = "cpu"
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
USE_LORENTZ_INVARIANT_FEATURES = True
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
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
if IS_XBB_TAGGED:
    N_CTX = 7 # the SIX types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
else:
    N_CTX = 6 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet'}
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
batch_size = 100
# This first one has removed events that aren't validly reconstructed (which might be interesting to look at)
DATA_PATH='/data/atlas/baines/20250314v1_AppliedRecoNN_WithRecoMasses_30_MetCut_RemovedUncertainTruth_WithTagInfo_KeepAllOldSelIncludingNegative/'
# This one doesn't contain all the background as far as I can tell. Might need to add this at some point
DATA_PATH=f'/data/atlas/baines/20250313v1_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'

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
    if (int(dsid) > 500000) and (int(dsid) < 600000):
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
)
print(val_dataloader.get_total_samples())

# %%


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
            HAS_MLP = False
        elif 0: # Without MLPs, 1 layer
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-124405_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=1, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
            HAS_MLP = False
        elif 0: # Without MLPs, 2 layers
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-123425_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=2, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
            HAS_MLP = False
        elif 1: # Without MLPs, 3 layers
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-123446_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=3, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
            HAS_MLP = False
        elif 1: # Without MLPs, 4 layers
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250318-123605_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 200, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(include_mlp=False, num_attention_blocks=4, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
            HAS_MLP = False
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
    num_batches_to_process = 1
    # num_batches_to_process = len(val_dataloader)
    for batch_idx in range(num_batches_to_process):
        if ((batch_idx%10)==9):
            print(F"Processing batch {batch_idx}/{num_batches_to_process}")
        batch = next(val_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        outputs = model(x[...,:N_Real_Vars], types).squeeze()
        val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)
    
rv=val_metrics_MCWts.compute_and_log(1,'val', 0, 3, False, None)



# %%
val_dataloader._reset_indices()
batch = next(val_dataloader)
x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
outputs = model(x[...,:N_Real_Vars], types).squeeze()
n=10
padding_token = N_CTX-1
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
importlib.reload(MechInterpUtils)

COMBINE_LEPTONS_FOR_PLOTS=False
with torch.no_grad():
    val_dataloader._reset_indices()
    batch = next(val_dataloader)
    x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
    cache = MechInterpUtils.extract_all_activations(model, x[...,:N_Real_Vars], types)
    residuals = MechInterpUtils.get_residual_stream(cache)

    dla=MechInterpUtils.old_direct_logit_attribution(model, cache)
    ndla=MechInterpUtils.direct_logit_attribution(model, cache)
    apa=MechInterpUtils.analyze_attention_patterns(cache,0)
    ota=MechInterpUtils.analyze_object_type_attention(model, cache, types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS)
    ota_selfex=MechInterpUtils.analyze_object_type_attention(model, cache, types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS, exclude_self=True)

# %%
from IPython.display import display
import circuitsvis as cv
from utils import Get_PtEtaPhiM_fromXYZT
chosen_sample_index = 0
SHOW_ALL=False
reconstructed_object_types = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'large-R jet', 4: 'small-R jet', 5:'None'}
for layer in range(len(model.attention_blocks)):
    attention_pattern = cache[f'block_{layer}_attention']['attn_weights'][chosen_sample_index]
    objselection = (types[chosen_sample_index:chosen_sample_index+1]!=(N_CTX-1)).flatten() # This allows us to ignore the 'pad' tokens (ie the 'nothing' particles)
    if SHOW_ALL:
        my_names=[f"{reconstructed_object_types[types[chosen_sample_index, obj].item()]:10} (E={x[chosen_sample_index,obj,3]:.2f})(M={Get_PtEtaPhiM_fromXYZT(x[chosen_sample_index,obj,0], x[chosen_sample_index,obj,1], x[chosen_sample_index,obj,2], x[chosen_sample_index,obj,3], use_torch=True)[-1]:.2f})" for obj in range(attention_pattern.shape[2])]
    else:
        attention_pattern = attention_pattern[:,objselection][:,:,objselection]
        my_names=[f"{reconstructed_object_types[types[chosen_sample_index, objselection][obj].item()]:10} (E={x[chosen_sample_index,objselection,3][obj]:.2f})(M={Get_PtEtaPhiM_fromXYZT(x[chosen_sample_index,objselection,0][obj], x[chosen_sample_index,objselection,1][obj], x[chosen_sample_index,objselection,2][obj], x[chosen_sample_index,objselection,3][obj], use_torch=True)[-1]:.2f})" for obj in range(attention_pattern.shape[2])]
    display(
        cv.attention.attention_patterns(
            tokens=my_names,  # type: ignore
            attention=attention_pattern,
            # attention_head_names=[f"L0H{i}" for i in range(model.cfg.n_heads)],
        )
    )

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
                    plt.imshow(ota_selfex[f"block_{layer}"][f"head_{head}"])
                else:
                    plt.imshow(ota[f"block_{layer}"][f"head_{head}"])
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.xticks(list(types_dict.keys()), list(types_dict.values()), rotation=90, size=10)
                plt.yticks(list(types_dict.keys()), list(types_dict.values()), rotation=0, size=10)
                plt.title(f"Layer {layer}, Head {head}")
        plt.suptitle("Average type attention patterns (rows are queries, cols are keys)")
        plt.tight_layout()
        plt.savefig(f'tmp{selfex*"_selfex"}.pdf')
        plt.show()


# %%
MechInterpUtils.current_attn_detector(model, cache)

# %%
importlib.reload(MechInterpUtils)
# MechInterpUtils.angular_separation_detector(model, cache, x, types, padding_token)
if 0:
    ah, scores = MechInterpUtils.angular_separation_detector_split_by_type(model, cache, x, types, padding_token, layers=[2], heads=[1], query_types=[0,1,2], key_types=[3])
elif 0:
    ah, scores = MechInterpUtils.angular_separation_detector_split_by_type(model, cache, x, types, padding_token, layers=[2], heads=[1], query_types=[2], key_types=[0, 1])
elif 0:
    ah, scores = MechInterpUtils.angular_separation_detector_split_by_type(model, cache, x, types, padding_token, layers=None, heads=None, query_types=[2], key_types=[0, 1])

# %%
importlib.reload(MechInterpUtils)
for obj_type in [0,1,2]:
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2)", include_mlp=HAS_MLP, include_direct_from_embedding=True)

# %%
for obj_type in [0,1,2]:
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [0], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to neither particle H+ decay product (class 0)", include_mlp=HAS_MLP)

# %%
truth_inclusion_mapping = {
    0:"neither particle",
    1:"SM Higgs",
    2:"W boson",
}
for true_incl in range(3):
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [3], title=f"Logit attribution for predicting reco large-R jets, truth-matched\nto be part of {truth_inclusion_mapping[true_incl]} (class {true_incl}) from H+", include_mlp=HAS_MLP)


# %%
for true_incl in range(3):
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [4], title=f"Logit attribution for predicting reco small-R jets, truth-matched\nto be part of {truth_inclusion_mapping[true_incl]} (class {true_incl}) from H+", include_mlp=HAS_MLP)



# %%
# %%
# %%
# %%
# %%
# %%

val_dataloader.N_Real_Vars_To_Return
model.object_net
# %%
importlib.reload(MechInterpUtils)
results = MechInterpUtils.main_sae_analysis(model, val_dataloader, N_Real_Vars, device=device)
# %%
