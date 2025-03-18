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
saveDir = "output/" + timeStr  + "_TrainingOutput/"
os.makedirs(saveDir)
print(saveDir)


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
batch_size = 1000
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
                    'post_attention': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_p),
                    )
                }) for _ in range(num_attention_blocks)
            ])
            # Final classification layers
            self.classifier = nn.Sequential(
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
                residual = identity + attention_output
                identity = residual
                # Post-attention processing
                mlp_output = block['post_attention'](residual)
                object_features = identity + mlp_output
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
        elif 0:
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250316-111909_TrainingOutput/models/0/chkpt19_109700.pth"
            model_cfg = {'d_model': 152, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariable(num_attention_blocks=5, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
    elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":
        if 1:
            modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250317-194149_TrainingOutput/models/0/chkpt9_54850.pth"
            model_cfg = {'d_model': 152, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
            model = DeepSetsWithResidualSelfAttentionVariableTrueSkip(num_attention_blocks=5, num_classes=3, input_dim=N_Real_Vars, hidden_dim=model_cfg['d_model'],  dropout_p=0.1,  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
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
    num_batches_to_process = 10
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

    dla=MechInterpUtils.direct_logit_attribution(model, cache)
    apa=MechInterpUtils.analyze_attention_patterns(cache,0)
    ota=MechInterpUtils.analyze_object_type_attention(model, cache, types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS)

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

if 0:
    for layer in range(len(model.attention_blocks)):
        plt.figure(figsize=(6,3))
        for head in range(model.attention_blocks[layer].self_attention.num_heads):
            plt.subplot(1,model.attention_blocks[layer].self_attention.num_heads,head+1)
            plt.imshow(ota[f"block_{layer}"][f"head_{head}"])
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks(list(types_dict.keys()), list(types_dict.values()), rotation=90, size=10)
            plt.yticks(list(types_dict.keys()), list(types_dict.values()), rotation=0, size=10)
        plt.suptitle(f"Layer: {layer}")
        plt.tight_layout()
        plt.show()
else:
    plt.figure(figsize=(6,15))
    for layer in range(len(model.attention_blocks)):
        for head in range(model.attention_blocks[layer].self_attention.num_heads):
            plt.subplot(len(model.attention_blocks),model.attention_blocks[layer].self_attention.num_heads,layer*model.attention_blocks[layer].self_attention.num_heads+head+1)
            plt.imshow(ota[f"block_{layer}"][f"head_{head}"])
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks(list(types_dict.keys()), list(types_dict.values()), rotation=90, size=10)
            plt.yticks(list(types_dict.keys()), list(types_dict.values()), rotation=0, size=10)
            plt.title(f"Layer {layer}, Head {head}")
    plt.suptitle("Average type attention patterns (rows are queries, cols are keys)")
    plt.tight_layout()
    plt.savefig('tmp.pdf')
    plt.show()


# %%
MechInterpUtils.current_attn_detector(model, cache)

# %%
# MechInterpUtils.close_in_phi_detector(model, cache, x)
# MechInterpUtils.close_in_eta_detector(model, cache, x)
# MechInterpUtils.close_in_R_detector(model, cache, x)


# %%
# import einops
# attention_pattern = cache[f"block_{layer}_attention"]["attn_weights"][:,head]
# # take avg of diagonal elements
# _, Eta, phi, _ = Get_PtEtaPhiM_fromXYZT(x[...,0], x[...,1], x[...,2], x[...,3], use_torch=True)
# deltaPhisOG = (einops.rearrange(phi, 'batch object -> batch object 1') - einops.rearrange(phi, 'batch object -> batch 1 object'))
# deltaEtasOG = (einops.rearrange(Eta, 'batch object -> batch object 1') - einops.rearrange(Eta, 'batch object -> batch 1 object'))

# if 1:
#     deltaPhis = deltaPhisOG
#     plt.hist(1/torch.pi*deltaPhis.flatten())
#     plt.show()

#     deltaPhis = torch.remainder(deltaPhisOG, 2*torch.pi)
#     plt.hist(1/torch.pi*deltaPhis.flatten())
#     plt.show()

#     deltaPhis = torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi
#     plt.hist(1/torch.pi*deltaPhis.flatten())
#     plt.show()

#     deltaPhis = torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi)
#     plt.hist(1/torch.pi*deltaPhis.flatten())
#     plt.show()

#     deltaPhis = torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi
#     plt.hist(1/torch.pi*deltaPhis.flatten())
#     plt.show()

# deltaPhis = torch.abs(torch.remainder(torch.remainder(deltaPhisOG, 2*torch.pi) + torch.pi, 2*torch.pi) - torch.pi)
# plt.hist(1/torch.pi*deltaPhis.flatten())
# plt.show()


# deltaEtas = deltaEtasOG.abs()

# object_types = types
# object_counts = (object_types!=padding_token).sum(dim=-1).unsqueeze(-1)
# (object_types!=padding_token).to(float)/object_counts
# valid_object = ((object_types!=padding_token).unsqueeze(-1) & (object_types!=padding_token).unsqueeze(1))
# non_diagonal = (~(torch.eye(object_types.shape[1]).to(bool))).unsqueeze(0)
# consider_for_delta_phi = valid_object & non_diagonal
# # flat_consider_for_delta_phi = einops.rearrange(consider_for_delta_phi, 'batch query key -> batch (query key)')
# flat_consider_for_delta_phi = einops.rearrange(consider_for_delta_phi, 'batch query key -> (batch query key)')
# flat_attention = einops.rearrange(cache[f"block_{layer}_attention"]["attn_weights"][:,head],'batch query key -> (batch query key)')
# flat_delta_phi = einops.rearrange(deltaPhis,'batch query key -> (batch query key)')
# flat_delta_eta = einops.rearrange(deltaEtas,'batch query key -> (batch query key)')


# # %%

# from gplearn.genetic import SymbolicRegressor
# from gplearn.functions import make_function
# # Create and fit the symbolic regressor
# X = np.concat([flat_delta_phi[flat_consider_for_delta_phi].cpu().numpy().reshape(-1,1),flat_delta_eta[flat_consider_for_delta_phi].cpu().numpy().reshape(-1,1)], axis=1)
# y = flat_attention[flat_consider_for_delta_phi].cpu().numpy()
# y = X[:,1] ** 8 + X[:, 0]*4.3
# MAX_ntrain=1000
# X = X[:MAX_ntrain]
# y = y[:MAX_ntrain]

# # assert(False)

# if 0:
#     # Define a safe power function
#     def safe_pow(x1, x2):
#         with np.errstate(over='ignore'):
#             return np.where((x1 >= 0) & (np.abs(x2) < 10), np.power(x1, x2), 1)

#     # Register the safe power function
#     safe_pow_func = make_function(function=safe_pow,
#                                 name='safe_pow',
#                                 arity=2)

# elif 0:
#     def _protected_exponent(x1): # From https://github.com/trevorstephens/gplearn/issues/49
#         with np.errstate(over='ignore'):
#             return np.where(np.abs(x1) < 100, np.exp(x1), 9999999.)

#     def _protected_exp(x1): # From https://github.com/trevorstephens/gplearn/issues/49
#         """Closure of exp for zero arguments."""
#         with np.errstate(divide='ignore', invalid='ignore'):
#             return np.where(np.abs(x1) > 0.001, np.exp(np.abs(x1)), 1.)
        
#         pexp = make_function(function=_protected_exp, name='exp', arity=1)


#     def _protected_power(x1, x2): # Adapted https://github.com/trevorstephens/gplearn/issues/49
#         """Closure of power for zero arguments."""
#         with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
#             a = np.where((x1 >= 0) & (np.abs(x2) < 100), np.power(x1, x2), 1)
#             max_val=99999999
#             a[~(x1 >= 0)] = 0
#             a[~(np.isfinite(a))] = max_val
#             a[~(np.abs(x2) < 100)] = max_val
#             return a
        
#     safe_pow_func = make_function(function=_protected_power, name='safe_pow', arity=2)
# else:
#     def _protected_power(x1, x2):
#         """
#         A safer implementation of power function to avoid overflow and domain errors.
#         """
#         with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
#             # Convert inputs to float to avoid integer overflow
#             x1 = x1.astype(float)
#             x2 = x2.astype(float)
            
#             # Limit the range of x2 to avoid extreme exponents
#             x2 = np.clip(x2, -10, 10)
            
#             # Handle negative bases
#             sign = np.sign(x1)
#             x1 = np.abs(x1)
            
#             # Compute power, with a maximum output value
#             max_val = 1e10
#             result = np.where((x1 != 0) & (x2 != 0), np.minimum(np.power(x1, x2), max_val), 1)
            
#             # Restore sign for odd integer powers
#             result = np.where(np.mod(x2, 2) == 1, result * sign, result)
            
#             # Handle special cases
#             result = np.where(x1 == 0, 0, result)  # 0^x = 0 for any x != 0
#             result = np.where((x1 == 0) & (x2 == 0), 1, result)  # 0^0 = 1 by convention
            
#             return result

#     safe_pow_func = make_function(function=_protected_power, name='safe_pow', arity=2)


# est_gp = SymbolicRegressor(population_size=5000, generations=20, function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'sin', 'cos', 'tan'])
# est_gp = SymbolicRegressor(population_size=5000, generations=20, function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg'])
# est_gp = SymbolicRegressor(
#     population_size=5000,
#     generations=20,
#     # function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', safe_pow_func],
#     # function_set=['add', 'sub', 'mul', 'sqrt', 'log', 'abs', 'neg', safe_pow_func],
#     # function_set=['mul', safe_pow_func],
#     function_set=['mul', 'add', safe_pow_func],
#     p_crossover=0.7,
#     p_subtree_mutation=0.1,
#     p_hoist_mutation=0.05,
#     p_point_mutation=0.1,
#     max_samples=0.9,
#     verbose=1,
#     parsimony_coefficient=0.01,
#     random_state=0
# )
# est_gp.fit(X, y)
# # Print the best found expression
# print(est_gp._program)
# # Make predictions
# y_pred = est_gp.predict(X)
# # Calculate R-squared score
# r_squared = est_gp.score(X, y)
# print(f"R-squared score: {r_squared}")

# # attention_pattern = cache[f"block_{layer}_attention"]["attn_weights"][:,head]
# # attention_pattern.shape


# # %%
# from pysr import PySRRegressor

# est_pysr = PySRRegressor(
#     population_size=500,
#     niterations=20,
#     binary_operators=["+", "*", "^"],
#     unary_operators=[],
#     constraints={'^': (-1, 1)},
#     maxsize=20,
#     parsimony=0.01,
#     procs=0.9,
#     ncyclesperiteration=500,
#     verbosity=1,
#     random_state=0,
#     deterministic=True,
#     procs=0,
#     # optimize_probability=0.7,
#     model_selection='accuracy',
# )

# est_pysr.fit(X, y)

# # Print the best found expression
# print(est_pysr.sympy())

# # Make predictions
# y_pred = est_pysr.predict(X)

# # Calculate R-squared score
# r_squared = est_pysr.score(X, y)
# print(f"R-squared score: {r_squared}")


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
# %%
# %%

for obj_type in [0,1,2]:
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [3], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2)")

# %%
for obj_type in [0,1,2]:
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [0], [obj_type], title=f"Logit attribution for predicting reco {reconstructed_object_types[obj_type]}s, truth-matched to W boson (class 2)")

# %%
for true_incl in range(3):
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [3])


# %%
for true_incl in range(3):
    _=MechInterpUtils.plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [4])




# %%
    














# %%
import einops
def direct_logit_attribution_tmp(model, 
                                 cache, 
                                 x, 
                                 truth_inclusion,
                                #  pred_inclusion, # Can get this from logits
                                 object_types, 
                                 padding_token,
                                 true_incl,
                                 type_incl,
                                 class_idx=None, 
                                 top_k=5,
                                 scale_z=None,
                                ):
    """
    Perform direct logit attribution to identify which components contribute most to classification.
    
    Args:
        model: The neural network model
        cache: ActivationCache with model activations
        class_idx: Class index to analyze (0=Neither, 1=Higgs, 2=W)
        top_k: Number of top contributors to return
        
    Returns:
        Dict of attribution scores by component
    """
    logits = cache["output"]
    batch_size, num_objects, num_classes = logits.shape
    
    attributions = {}

    mask = ((torch.isin(truth_inclusion, torch.Tensor(true_incl))) & 
            (torch.isin(object_types, torch.Tensor(type_incl)))
            ).to(logits.device)

    # If no specific class is provided, use the predicted class
    if class_idx is None:
        print("Class index is None so just filling with the true class")
        # class_idx = logits.argmax(dim=-1)
        class_idx = (truth_inclusion==1) + (truth_inclusion>1)*2
        class_idx_label = True
        classifier_weights = model.classifier[0].weight[class_idx]  # shape: [hidden_dim]
    elif isinstance(class_idx, int):
        class_idx_label = class_idx
        class_idx = torch.full((batch_size, num_objects), class_idx, device=logits.device)
        # Get the classifier weights for the specified class
        classifier_weights = model.classifier[0].weight[class_idx]  # shape: [hidden_dim]
    elif isinstance(class_idx, tuple):
        class_idx_label=f"{class_idx[0]}<-{class_idx[1]}"
        class_idx_target = torch.full((batch_size, num_objects), class_idx[0], device=logits.device)
        class_idx_nontarget = torch.full((batch_size, num_objects), class_idx[1], device=logits.device)
        # Get the classifier weights for the specified class
        classifier_weights = model.classifier[0].weight[class_idx_target] - model.classifier[0].weight[class_idx_nontarget] # shape: [hidden_dim]
    



    # Compute attributions for each attention block
    for i in range(model.num_attention_blocks):
        # Get attention outputs
        attn_weights = cache[f"block_{i}_attention"]["attn_weights"]
        attn_output = cache[f"block_{i}_attention"]["attn_output"]
        
        
        # Calculate attribution per attention head
        num_heads = attn_weights.shape[1]
        head_attributions = []
        
        # Reshape attention output to separate heads
        # For standard nn.MultiheadAttention, we need to infer head dimension
        hidden_dim = attn_output.shape[-1]
        head_dim = hidden_dim // num_heads
        
        for h in range(num_heads):
            # Extract this head's contribution
            # This is approximate as we don't have direct access to per-head outputs
            head_slice = slice(h * head_dim, (h + 1) * head_dim)
            head_output = attn_output[..., head_slice]
            
            # Calculate attribution (dot product with classifier weights)
            head_attribution = torch.sum(head_output * classifier_weights[..., head_slice], dim=-1)
            head_attributions.append(head_attribution)
        
        attributions[f"block_{i}_attention"] = torch.stack(head_attributions, dim=0)
    
    # Aggregate attributions across batches for analysis
    aggregated = {}
    for key, attr in attributions.items():
        # Average across batch and objects
        aggregated[key] = einops.einsum(attr* mask, 'head batch object -> head') / einops.einsum(mask, 'batch object ->')
        # aggregated[key] = attr.mean(dim=(1, 2)).cpu().numpy()
    
    # Find top contributors
    all_attrs = []
    for key, attrs in aggregated.items():
        for i, attr in enumerate(attrs):
            all_attrs.append((f"{key}_head_{i}", attr))
    
    plt.figure(figsize=(5,4))
    conts = np.zeros((model.num_attention_blocks, num_heads))
    for i in range(model.num_attention_blocks):
        conts[i] = aggregated[f"block_{i}_attention"].detach().cpu()
    # plt.imshow(conts, cmap='RdBu')
    if scale_z is not None:
        plt.imshow(conts, cmap='PiYG', vmin=-scale_z, vmax=scale_z)
    else:
        plt.imshow(conts, cmap='PiYG')
    # plt.imshow(conts, cmap='RdBu_r')
    plt.title(f"Direct logit attribution for predicting class: {class_idx_label}\nAcross objects with true-assignment in {true_incl}, type in {type_incl}")
    plt.colorbar()
    plt.show()
        # for h in range(num_heads):
    #         plt.subplot(model.num_attention_blocks, num_heads, i*num_heads + h + 1)
    
    # Sort by absolute attribution
    all_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "top_components": all_attrs[:top_k],
        "all_attributions": attributions,
        "aggregated": aggregated
    }

truth_inclusion = x[:,...,-1]
# print(f"{truth_inclusion.shape}")
pred_inclusion=outputs.argmax(dim=-1)
# print(f"{pred_inclusion.shape}")

r=direct_logit_attribution_tmp(model, 
                             cache, 
                             x[...,:4], 
                             x[...,-1], 
                            #  outputs.argmax(-1), 
                             types, 
                             padding_token,
                             [3],
                             [2],
)
# %%

r=direct_logit_attribution_tmp(model, 
                             cache, 
                             x[...,:4], 
                             x[...,-1], 
                            #  outputs.argmax(-1), 
                             types, 
                             padding_token,
                             [3],
                             [2],
                             class_idx=2
)
# %%

for class_idx in [2, 1, 0, (2,1), (2,0)]:
    # for tt in [0,1,2]:
    for tt in [2]:
        r=direct_logit_attribution_tmp(model, 
                                    cache, 
                                    x[...,:4], 
                                    x[...,-1], 
                                    #  outputs.argmax(-1), 
                                    types, 
                                    padding_token,
                                    [3], # True is present in W boson
                                    [tt], # Type is neutrino, electron or muon
                                    class_idx=class_idx
        )


for _ in range(10):
    print('--------------------------------------------------------')

for class_idx in [0, 1, 2, (0,2)]:
    r=direct_logit_attribution_tmp(model, 
                                cache, 
                                x[...,:4], 
                                x[...,-1], 
                                #  outputs.argmax(-1), 
                                types, 
                                padding_token,
                                [0], # True is NOT present in W/SM Higgs
                                [2], # Type is neutrino
                                class_idx=class_idx
    )


for _ in range(10):
    print('--------------------------------------------------------')

for class_idx in [1, 0, 2, (1,0), (1,2)]:
    r=direct_logit_attribution_tmp(model, 
                                cache, 
                                x[...,:4], 
                                x[...,-1], 
                                #  outputs.argmax(-1), 
                                types, 
                                padding_token,
                                [1], # True is SM Higgs
                                [3], # Type is large-R jet
                                class_idx=class_idx
    )

for _ in range(10):
    print('--------------------------------------------------------')

for class_idx in [1, 0, 2, (1,0), (1,2)]:
    r=direct_logit_attribution_tmp(model, 
                                cache, 
                                x[...,:4], 
                                x[...,-1], 
                                #  outputs.argmax(-1), 
                                types, 
                                padding_token,
                                [1], # True is SM Higgs
                                [4], # Type is small-R jet
                                class_idx=class_idx
    )

for _ in range(10):
    print('--------------------------------------------------------')

for class_idx in [2, 0, 1, (2,0), (2,1)]:
    r=direct_logit_attribution_tmp(model, 
                                cache, 
                                x[...,:4], 
                                x[...,-1], 
                                #  outputs.argmax(-1), 
                                types, 
                                padding_token,
                                [2], # True is W
                                [4], # Type is small-R jet
                                class_idx=class_idx
    )

    
# %%
for _ in range(100):
    print("NEED TO ADD CODE TO EXTRACT MLP LAYERS")




true_class = [3] # True is NOT present in W/SM Higgs
type_incl = [2] # Type is neutrino
plot_logit_attributions(model, cache, x[...,-1], types, [3], [2])


# %%
for obj_type in [0,1,2]:
    _=plot_logit_attributions(model, cache, x[...,-1], types, [3], [obj_type])
for _ in range(5):
    print("---------------")

# for obj_type in [0,1,2]:
#     _=plot_logit_attributions(model, cache, x[...,-1], types, [2], [obj_type])
# for _ in range(5):
#     print("---------------")

# for obj_type in [0,1,2]:
#     _=plot_logit_attributions(model, cache, x[...,-1], types, [1], [obj_type])
# for _ in range(5):
#     print("---------------")

for obj_type in [0,1,2]:
    _=plot_logit_attributions(model, cache, x[...,-1], types, [0], [obj_type])
for _ in range(5):
    print("---------------")


# %%
for true_incl in range(3):
    _=plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [3])


# %%
for true_incl in range(3):
    _=plot_logit_attributions(model, cache, x[...,-1], types, [true_incl], [4])



# %%
