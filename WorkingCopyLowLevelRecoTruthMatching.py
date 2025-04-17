# %%
DRY_RUN=False
# %% # Load required modules
import time
ts = []
ts.append(time.time())

import numpy as np
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import torch
from datetime import datetime
from jaxtyping import Float
from typing import List
import einops
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# from utils import save_current_script#, decode_y_eval_to_info
from utils import decode_y_eval_to_info
from mynewdataloader import ProportionalMemoryMappedDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
import wandb
import torch.nn.functional as F
# from torchmetrics import Accuracy, AUC, ConfusionMatrix
# from torchmetrics import ConfusionMatrix
# from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import shutil
# from MyMetricsLowLevel import HEPMetrics, HEPLoss, init_wandb
from MyMetricsLowLevelRecoTruthMatching import HEPMetrics, HEPLoss, HEPLossWithEntropy, init_wandb
# from MyMetricsLowLevelRecoTruthMatchingTmp import HEPMetrics, HEPLoss, init_wandb
import functools

# %%
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")
saveDir = "output/" + timeStr  + "_TrainingOutput/"
os.makedirs(saveDir)
print(saveDir)
# Save the current script to the saveDir so we know what the training script was
def save_current_script(destination_directory):
    current_script_path = os.path.abspath(__file__)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    destination_path = os.path.join(destination_directory, os.path.basename(current_script_path))
    shutil.copy(current_script_path, destination_path)
    print(f"This script copied to: {destination_path}")
if DRY_RUN:
    pass
else:
    save_current_script('%s'%(saveDir))

# Some choices about the training process
# Assumes that the data has already been binarised
USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION = False # Bool. If true, we'll apply a loss penalty which encourages the attention weights to follow a distribution close to a specific entropy (can make this 0 for close to delta function ie 'pay attention to exactly one particle', log(2) for close to 'pay attention to exactly two particles', ...); aim of this is to make the model more interpretable
ATTENTION_OUTPUT_BOTTLENECK_SIZE = 1 # None or integer. If not None, then we will apply a linear reduction then expansion to the attention output (per head) to this integer, to reduce the dimensionality of data that can be passed around; aim of this is to make the model more interpretable
IS_CATEGORICAL = True
PHI_ROTATED = False
REMOVE_WHERE_TRUTH_WOULD_BE_CUT = True
MODEL_ARCH="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP_WITH_BOTTLENECK"
TAG_INFO_INPUT=True
# MODEL_ARCH="DEEPSETS_RESIDUAL_LONGCLASSIFIER"
# if not (MODEL_ARCH=="HYBRID_SELFATTENTION_GATED"):
if True:
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: # For testing new model architecture classes, probably run on cpu, since CUDA will just give difficult errors if there is some problem with the arch
    device = "cpu"
USE_LORENTZ_INVARIANT_FEATURES = True
TOSS_UNCERTAIN_TRUTH = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = False
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here (well, really in the prep data script) or not
SHUFFLE_OBJECTS = True
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
    N_Real_Vars=7 # px, py, pz, energy, tagInfo.  BE CAREFUL because this might change and if it does you ahve to rebinarise
else:
    N_Real_Vars=6 # px, py, pz, energy.  BE CAREFUL because this might change and if it does you ahve to rebinarise


# %%
# Set up stuff to read in data from bin file


batch_size = 256
DATA_PATH=f'/data/atlas/baines/tmp2_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired' + '_mHSel'*MH_SEL + '/'
DATA_PATH=f'/data/atlas/baines/tmp_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired' + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '/'
DATA_PATH=f'/data/atlas/baines/tmp_SingleXbbSelected' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '/'
DATA_PATH=f'/data/atlas/baines/tmp3_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '/'
DATA_PATH=f'/data/atlas/baines/tmp6_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '/'
DATA_PATH=f'/data/atlas/baines/tmp7_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'
DATA_PATH=f'/data/atlas/baines/20250313v1_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'
DATA_PATH=f'/data/atlas/baines/20250314v1_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5_LowLevelRecoTruthMatching' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'
DATA_PATH=f'/data/atlas/baines/20250321v1_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + f'_WithRecoMasses_{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'

if NORMALISE_DATA:
    means = np.load(f'{DATA_PATH}mean.npy')[1:]
    stds = np.load(f'{DATA_PATH}std.npy')[1:]
else:
    means = None
    if SCALE_DATA:
        stds = np.ones(N_Real_Vars)
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
        # if (int(dsid)!=510115) and (int(dsid)!=510116) and (int(dsid)!=510117):
        # if int(dsid)!=510120:
        # if int(dsid)<=510122:
        # if int(dsid)>=510117:
        if True:
            memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    else:
        pass # Skip because we don't train the reco on bkg
    if (int(dsid) > 500000) and (int(dsid) < 600000):
        memmap_paths_val[int(dsid)] = DATA_PATH+file_name
    else:
        print("For now, don't include the background even in the val set, but later we might want to add some backgorund performance logging to the Metric tracking code")
n_splits=2
validation_split_idx=0
train_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_train,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars,
                 N_Real_Vars_To_Return=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 signal_only=True,
                 has_eventNumbers=True,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_val,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars,
                 N_Real_Vars_To_Return=N_Real_Vars,
                 class_proportions = None,
                #  batch_size=64*8*64*8,
                #  batch_size=64*8*64,
                 batch_size=batch_size,
                 device=device,
                 is_train=False,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 signal_only=True,
                 has_eventNumbers=True,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
print(train_dataloader.get_total_samples())
print(val_dataloader.get_total_samples())
# assert(False) # Need to check if the weighting is correct - it seemed likely that the sum of training weights for signal was not the same as for background?
# %%
batch = next(train_dataloader)

# %%
if 0:
    plt.figure(figsize=(4,4))
    nbins=20
    bins=np.arange(nbins+1)/nbins*200e3+50e3
    plt.hist(batch['mH'][((batch['dsids']<500000)|(batch['dsids']>600000))].cpu(),histtype='step',bins=bins, label='Bkg')
    plt.hist(batch['mH'][~((batch['dsids']<500000)|(batch['dsids']>600000))].cpu(),histtype='step',bins=bins, label='Sig')
    plt.show()
    plt.figure(figsize=(4,4))
    nbins=20
    bins=np.arange(nbins+1)/nbins*200e3+50e3
    plt.hist(batch['mH'][batch['y'].argmax(dim=-1)==0].cpu(),histtype='step',bins=bins, label='Bkg')
    plt.hist(batch['mH'][batch['y'].argmax(dim=-1)!=0].cpu(),histtype='step',bins=bins, label='Sig')
    plt.show()
# plt.hist(batch['mH'][batch['dsids']==510124],histtype=='step')

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

if MODEL_ARCH=="DEEPSETS_BASIC":
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
elif MODEL_ARCH=="TFLENS_DEEPSETS_RESIDUAL_VARIABLE":
    class TfLensDeepsetsResidualSelfAttention(HookedTransformer):
        def __init__(self, is_categorical, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, embedding_size=32, num_layers=3, **kwargs):
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
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":
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
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP_WITH_BOTTLENECK":
    class DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(nn.Module):
        def __init__(self, bottleneck_attention=None, feature_set=['pt', 'eta', 'phi', 'm', 'tag'], num_classes=3, hidden_dim=256, num_heads=4, dropout_p=0.0, embedding_size=32, num_attention_blocks=3, include_mlp=True, hidden_dim_mlp=None):
            super().__init__()
            self.bottleneck_attention = bottleneck_attention
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
elif MODEL_ARCH=="GRAPH_NN": # Not finished
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
elif MODEL_ARCH=="SINGLE_SELFATTENTION":
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
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_LONGCLASSIFIER":
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
elif MODEL_ARCH=="HYBRID_SELFATTENTION_GATED":
    assert(False) # Need to turn this into a classifier per object
    class HybridAttentionDeepSets(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, num_heads=4, embedding_size=32, dropout_p=0.0):
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
            
            # Self-attention layer for object interactions (with fewer heads)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            
            # Gated attention mechanism
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Sigmoid()
            )
            
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh()
            )
            
            # Attention integration layer
            self.attention_integration = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # Combine self-attention and gated attention
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
            # batch_size, num_objects, feature_dim = x.shape
            # Assuming last feature is object type
            if USE_LORENTZ_INVARIANT_FEATURES:
                object_features[...,:4] = self.invariant_features(object_features[...,:4])
            
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(object_types)
            combined = torch.cat([object_features, type_emb], dim=-1)
            
            # Process each object
            object_features = self.object_net(combined)
            
            # Apply self-attention to model interactions between objects
            self_attention_output, _ = self.self_attention(
                object_features, object_features, object_features
            )
            
            # Apply gated attention mechanism
            gates = self.gate_net(object_features)
            attention_values = self.attention_net(object_features)
            gated_features = gates * attention_values
            
            # Create two pooled representations
            # 1. Self-attention based pooling
            pooled_self_attention = torch.sum(self_attention_output, dim=1)
            
            # 2. Gated attention based pooling
            pooled_gated_attention = torch.sum(gated_features, dim=1)
            
            # Combine both attention mechanisms
            combined_features = torch.cat([pooled_self_attention, pooled_gated_attention], dim=1)
            integrated_features = self.attention_integration(combined_features)
            
            return self.classifier(integrated_features)
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X3":
    # assert(False) # Need to turn this into a classifier per object
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
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X2":
    # assert(False) # Need to turn this into a classifier per object
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
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL":
    # assert(False) # Need to turn this into a classifier per object
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
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION":
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
elif MODEL_ARCH=="DEEPSETS":
    assert(False) # Need to turn this into a classifier per object
    class DeepSetsWithGatedAttention(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, dropout_p=0.0, embedding_size=32):
            super().__init__()
            if USE_LORENTZ_INVARIANT_FEATURES:
                self.invariant_features = LorentzInvariantFeatures()
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)  # 5 object types
            # Initial per-object processing
            self.object_net = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # +32 for type embedding
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Gated attention mechanism
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            # Final classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                # nn.Dropout(dropout_p),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        def forward(self, object_features, types):
            # Get type embeddings and combine with features
            type_emb = self.type_embedding(types)
            if USE_LORENTZ_INVARIANT_FEATURES:
                object_features[...,:4] = self.invariant_features(object_features[...,:4])
            combined = torch.cat([object_features, type_emb], dim=-1)
            # Process each object
            object_features = self.object_net(combined)
            # Apply gated attention
            gates = self.gate_net(object_features)
            attention = self.attention_net(object_features)
            gated_features = gates * attention
            # Permutation-invariant pooling
            pooled = torch.sum(gated_features, dim=1)
            return self.classifier(pooled)
elif MODEL_ARCH=="PARTICLE_FLOW":
    assert(False) # Need to turn this into a classifier per object
    class LorentzInvariantParticleFlow(nn.Module):
        def __init__(self, input_dim=5, num_classes=3, hidden_dim=256, dropout_p=0.0, embedding_size=32):
            super().__init__()
            
            self.invariant_features = LorentzInvariantFeatures()
            
            # Object type embedding
            self.type_embedding = nn.Embedding(N_CTX, embedding_size)

            # Initial node feature network
            self.node_init = nn.Sequential(
                nn.Linear(input_dim + embedding_size, hidden_dim),  # 4 invariant features + tagInfo + type embedding
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Edge network for computing pairwise features
            self.edge_network = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),  # 4 invariant features + tagInfo per object
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Node update network
            self.node_network = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),  # edge features + invariant features
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim//2, num_classes)
            )
        
        def forward(self, object_features, types):
            batch_size, n_particles, _ = object_features.shape
            
            # Split input into components
            four_momenta = object_features[..., :4]
            tag_info = object_features[..., 4:5]
            obj_type = types.long()
            
            # Calculate Lorentz invariant features
            invariant_features = self.invariant_features(four_momenta)
            type_emb = self.type_embedding(obj_type)
            
            # Combine invariant features with tag info and type embeddings
            node_features = torch.cat([invariant_features, tag_info, type_emb], dim=-1)
            
            # Initialize node representations
            node_features = self.node_init(node_features)
            
            # Compute pairwise interactions
            node_features_i = node_features.unsqueeze(2).expand(-1, -1, n_particles, -1)
            node_features_j = node_features.unsqueeze(1).expand(-1, n_particles, -1, -1)
            edge_input = torch.cat([node_features_i, node_features_j], dim=-1)
            
            # Process edges
            edge_features = self.edge_network(edge_input)
            edge_features = torch.mean(edge_features, dim=2)  # Average over neighbors
            
            # Update node features
            node_input = torch.cat([edge_features, node_features], dim=-1)
            final_features = self.node_network(node_input)
            
            # Global pooling
            pooled = torch.mean(final_features, dim=1)
            
            return self.classifier(pooled)
elif MODEL_ARCH=="TRANSFORMER":
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
else:
    assert(False)


# %%
# Create a new model
models = {}
fit_histories = {}
model_n = 0

num_blocks_variable=6
num_clasifierlayers_variable=8
if MODEL_ARCH=="TRANSFORMER":
    # Create the model with the desired properties
    model_cfg = HookedTransformerConfig(
        # normalization_type='LN',
        normalization_type='LN',
        d_model=128,
        d_head=8,
        n_layers=3,
        n_heads=3,
        n_ctx=N_CTX, # Max number of types of object per event + 1 because we want a dummy row in the embedding matrix for non-existing particles
        d_vocab=4+int(TAG_INFO_INPUT), # Number of inputs per object
        d_vocab_out=1,  # 1 because we're doing binary classification
        d_mlp=256,
        attention_dir="bidirectional",  # defaults to "causal"
        act_fn="relu",
        use_attn_result=True,
        device=str(device),
        use_hook_tokens=True,
    )
elif MODEL_ARCH=="DEEPSETS" or MODEL_ARCH=="PARTICLE_FLOW":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":32}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":32, "num_heads":1}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":10, "num_heads":2}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X2":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":10, "num_heads":4}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X3":
    model_cfg = {'d_model': 300, 'dropout_p': 0.2, "embedding_size":10, "num_heads":4}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":
    model_cfg = {'include_mlp':False, 'd_model': 300, 'd_mlp': 300, 'num_blocks':num_blocks_variable, 'dropout_p': 0.0, "embedding_size":10, "num_heads":4}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP_WITH_BOTTLENECK":
    model_cfg = {'include_mlp':False, 'd_model': 200, 'd_mlp': 300, 'num_blocks':num_blocks_variable, 'dropout_p': 0.0, "embedding_size":10, "num_heads":4}
elif MODEL_ARCH=="SINGLE_SELFATTENTION":
    model_cfg = {'layer_norm_input':True, 'skip_connection':True, 'include_mlp':False, 'd_model': 400, 'd_mlp': 200, 'num_blocks':num_blocks_variable, 'dropout_p': 0.0, "embedding_size":10, "num_heads":4}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE":
    model_cfg = {'d_model': 152, 'num_blocks':num_blocks_variable, 'dropout_p': 0.1, "embedding_size":10, "num_heads":2}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_LONGCLASSIFIER":
    model_cfg = {'num_classifyer_layers':num_clasifierlayers_variable, 'd_model': 200, 'num_blocks':num_blocks_variable, 'dropout_p': 0.0, "embedding_size":10, "num_heads":2}
elif MODEL_ARCH=="HYBRID_SELFATTENTION_GATED":
    model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":32, "num_heads":1}
elif MODEL_ARCH=="TFLENS_DEEPSETS_RESIDUAL_VARIABLE":
    model_cfg = {'d_model': 256, 'dropout_p': 0.0, "embedding_size":10, "num_heads":4, "num_layers":num_blocks_variable}
elif MODEL_ARCH=="DEEPSETS_BASIC":
    model_cfg = {'num_blocks':num_blocks_variable, 'd_model': 300, 'dropout_p': 0.1, "embedding_size":16, "num_heads":4}
else:
    assert(False)

if IS_CATEGORICAL:
    num_classes=3
else:
    num_classes=1

if MODEL_ARCH=="HYBRID_SELFATTENTION_GATED":
    models[model_n] = {'model' : HybridAttentionDeepSets(input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads']).to(device)}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttention(num_classes=num_classes, input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X3":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttentionTriple(num_classes=num_classes, input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttentionVariableTrueSkip(feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, hidden_dim_mlp=model_cfg['d_mlp'], include_mlp=model_cfg['include_mlp'], num_attention_blocks=model_cfg['num_blocks'], hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP_WITH_BOTTLENECK":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttentionVariableTrueSkipBottleneck(bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, hidden_dim_mlp=model_cfg['d_mlp'], include_mlp=model_cfg['include_mlp'], num_attention_blocks=model_cfg['num_blocks'], hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="SINGLE_SELFATTENTION":
    models[model_n] = {'model' : SingleSelfAttentionNet(layer_norm_input=model_cfg['layer_norm_input'], skip_connection=model_cfg['skip_connection'], hidden_dim_mlp=model_cfg['d_mlp'], include_mlp=model_cfg['include_mlp'], input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_VARIABLE":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttentionVariable(num_attention_blocks=model_cfg['num_blocks'], input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_RESIDUAL_LONGCLASSIFIER":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttentionVariableLongclassifier(num_classifier_layers=num_clasifierlayers_variable, num_attention_blocks=model_cfg['num_blocks'], input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION_RESIDUAL_X2":
    models[model_n] = {'model' : DeepSetsWithResidualSelfAttentionDouble(num_classes=num_classes, input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_SELFATTENTION":
    models[model_n] = {'model' : DeepSetsWithSelfAttention(num_classes=num_classes, input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS":
    models[model_n] = {'model' : DeepSetsWithGatedAttention(input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p']).to(device)}
elif MODEL_ARCH=="PARTICLE_FLOW":
    models[model_n] = {'model' : LorentzInvariantParticleFlow(input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p']).to(device)}
elif MODEL_ARCH=="TRANSFORMER":
    if IS_CATEGORICAL:
        models[model_n] = {'model' : MyHookedTransformer(model_cfg, IS_CATEGORICAL, num_classes=num_classes).to(device)}
elif MODEL_ARCH=="TFLENS_DEEPSETS_RESIDUAL_VARIABLE":
    models[model_n] = {'model' : TfLensDeepsetsResidualSelfAttention(IS_CATEGORICAL, num_layers=model_cfg['num_layers'], input_dim=4+int(TAG_INFO_INPUT), num_classes=num_classes, hidden_dim=model_cfg['d_model'], num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
elif MODEL_ARCH=="DEEPSETS_BASIC":
    models[model_n] = {'model' : DeepSetsBasic(num_attention_blocks=model_cfg['num_blocks'], input_dim=4+int(TAG_INFO_INPUT), hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)}
else:
    assert(False)
print(sum(p.numel() for p in models[model_n]['model'].parameters()))

# %%
for p in models[model_n]['model'].named_parameters():
    print(f"{p[0]:50s}: {p[1].numel():10d}")
print(sum(p.numel() for p in models[model_n]['model'].parameters()))


# %%
if (MODEL_ARCH=="TRANSFORMER") or (MODEL_ARCH=="TFLENS_DEEPSETS_RESIDUAL_VARIABLE"):
    def add_perma_hooks_to_mask_pad_tokens(
        model: HookedTransformer
    ) -> HookedTransformer:
        # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
        def cache_padding_tokens_mask(tokens: Float[Tensor, "batch object d_input"], hook: HookPoint) -> None:
            # print("Caching padding tokens!")
            hook.ctx["padding_tokens_mask"] = einops.rearrange(torch.all(tokens==0, dim=-1), "b sK -> b 1 1 sK")

        # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
        def apply_padding_tokens_mask(
            attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
            hook: HookPoint,
        ) -> None:
            attn_scores.masked_fill_(model.hook_dict["hook_ObjectInputs"].ctx["padding_tokens_mask"], -1e5)
            if hook.layer() == model.cfg.n_layers - 1:
                del model.hook_dict["hook_ObjectInputs"].ctx["padding_tokens_mask"]

        # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
        for name, hook in model.hook_dict.items():
            if name == "hook_ObjectInputs":
                hook.add_perma_hook(cache_padding_tokens_mask)  # type: ignore
            elif name.endswith("attn_scores"):
                hook.add_perma_hook(apply_padding_tokens_mask)  # type: ignore

        return model

    def dropout_hook(
        resid: Float[Tensor, "batch object d_model"],
        hook: HookPoint,
        p=0.1,
        v=-1e5,
    ) -> None:
        resid.masked_fill_((torch.rand(resid.shape) < p).to(resid.device), v)

    models[model_n]['model'].reset_hooks(including_permanent=True)
    models[model_n]['model'] = add_perma_hooks_to_mask_pad_tokens(models[model_n]['model'])


class_weights = [1 for _ in range(N_TARGETS)]
labels = ['Bkg', 'Lep', 'Had'] # truth==0 is bkg, truth==1 is leptonic decay, truth==2 is hadronic decay
class_weights_expanded = einops.repeat(torch.Tensor(class_weights), 't -> batch t', batch=batch_size).to(device)



# SHOULD CHANGE WEIGHT DECAY BACK (IT WAS 1e-5 before)
# criterion = nn.CrossEntropyLoss(reduction='none')  # 'none' to handle sample weights
# for epoch in range(num_epochs):
#     dataloader._reset_indices()
    
#     # Training
#     models[model_n]['model'].train()
#     num_steps = len_train_dataloader-1
#     for n_step in range(num_steps): #TODO currently we don't handle the last batch, fix this
#         batch = next(dataloader)
#         x, y, w, types, y_eval, mWH_qqbb, mWH_lvbb = batch['x'], batch['y'], batch['w'], batch['types'], batch['y_eval'], batch['mWH_qqbb'], batch['mWH_lvbb']
#         class_weights_expanded = einops.repeat(torch.Tensor(class_weights), 't -> l t', l=y.shape[0]).to(device)
#         outputs = models[model_n]['model'](x, types)
#         class_loss = (criterion(outputs, y) * w).mean()
#         optimizer.zero_grad()
#         class_loss.backward()
#         optimizer.step()
#         if (n_step % 10) == 0:
#             print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, num_steps,class_loss.item()))

# %%
from utils import basic_lr_scheduler
if 0:
    n_epochs = 100
    plt.plot([basic_lr_scheduler(i, 1, 1e-3, n_epochs=n_epochs, log=False) for i in range(n_epochs)])
    plt.plot([basic_lr_scheduler(i, 1, 1e-3, n_epochs=n_epochs, log=True) for i in range(n_epochs)])
    plt.yscale('log')
    plt.show()
# [basic_lr_scheduler(i, 1, 1e-3, n_epochs=n_epochs) for i in range(n_epochs)]


# %%
model, train_loader, val_loader = models[model_n]['model'], train_dataloader, val_dataloader
num_epochs = 30
# log_interval = 100
log_interval = int(50e3/batch_size)
# longer_log_interval = log_interval*10
longer_log_interval = 100000000000
SAVE_MODEL_EVERY = 2
name_mapping = {"DEEPSETS":"DS", 
                "HYBRID_SELFATTENTION_GATED":"DSSAGA", 
                "DEEPSETS_SELFATTENTION":"DSSA", 
                "DEEPSETS_SELFATTENTION_RESIDUAL":"DSSAR", 
                "DEEPSETS_SELFATTENTION_RESIDUAL_X2":"DSSAR2", 
                "DEEPSETS_SELFATTENTION_RESIDUAL_X3":"DSSAR3", 
                "DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":f"DSSARVTS{num_blocks_variable}",
                "DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP_WITH_BOTTLENECK":f"DSSARVTSBN{num_blocks_variable}",
                "SINGLE_SELFATTENTION":f"SSA",
                "DEEPSETS_RESIDUAL_VARIABLE":f"DSSARV{num_blocks_variable}",
                "DEEPSETS_RESIDUAL_LONGCLASSIFIER":f"DSSAR_{num_blocks_variable}B_{num_clasifierlayers_variable}CL",
                "PARTICLE_FLOW":"PF",
                "TRANSFORMER":"TF",
                "TFLENS_DEEPSETS_RESIDUAL_VARIABLE":f"TLDSSARV{num_blocks_variable}",
                "DEEPSETS_BASIC":f"DSB",
                }
config = {
        "learning_rate": 3e-4,
        "learning_rate_low": 5e-7,
        "learning_rate_log_decay":True,
        "architecture": "PhysicsTransformer",
        "dataset": "ATLAS_ChargedHiggs",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "wandb":True,
        "name":"_"+timeStr+"_LowLevel_"+name_mapping[MODEL_ARCH],
        "weight_decay":1e-6,
    }
if DRY_RUN:
    config["wandb"] = False
optimizer = torch.optim.Adam(models[model_n]['model'].parameters(), lr=1e-4, weight_decay=config['weight_decay'])
if config['wandb']:
    init_wandb(config)
    # wandb.watch(model, log_freq=100)

# %%
if 0: # 
    print("WARNING: You are starting from a semi-pre-trained model state")
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250302-224613_TrainingOutput/models/0/chkpt74_414975.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
    models[model_n]['model'].load_state_dict(loaded_state_dict)


# %%
if not USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION:
    criterion = HEPLoss(is_categorical=IS_CATEGORICAL, apply_correlation_penalty=False, alpha=1.0, apply_valid_penalty=False, valid_penalty_weight=1.0)
else:
    criterion = HEPLossWithEntropy(entropy_loss=True, entropy_weight=1e-3, target_entropy=np.log(2), is_categorical=IS_CATEGORICAL, apply_correlation_penalty=False, alpha=1.0, apply_valid_penalty=False, valid_penalty_weight=1.0)
    # criterion = HEPLossWithEntropy(entropy_loss=True, entropy_weight=1e-3, target_entropy=0, is_categorical=IS_CATEGORICAL, apply_correlation_penalty=False, alpha=1.0, apply_valid_penalty=False, valid_penalty_weight=1.0)
train_metrics = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
train_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics_MCWts = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
global_step = 0
total_train_samples_processed = 0
train_loader._reset_indices()
orig_len_train_dataloader=len(train_loader)
num_lr_steps = num_epochs*orig_len_train_dataloader
from MechInterpUtils import run_with_cache_and_singleAttention, run_with_cache_and_bottleneck, run_with_cache_and_minAttention
for epoch in range(num_epochs):
    # Update learning rate based on the cosine scheduler
    train_loader._reset_indices()
    val_loader._reset_indices()
    train_metrics.reset()
    train_metrics_MCWts.reset()
    val_metrics.reset()
    val_metrics_MCWts.reset()
    model.train()
    n_step = 0
    orig_len_train_dataloader=len(train_loader)
    train_loss_epoch = 0
    sum_weights_epoch = 0
    for batch_idx in range(orig_len_train_dataloader):
        if ((batch_idx%10)==0):
            new_lr = basic_lr_scheduler(batch_idx + epoch*orig_len_train_dataloader, config['learning_rate'], config['learning_rate_low'], num_lr_steps, config["learning_rate_log_decay"], warmup_steps=100, warmup_rate=3e-3) # Or in theory could use global step
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            if ((batch_idx%1000)==0):
                print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {new_lr:.6e}")
    # Training phase
        n_step+=1
        global_step += 1
        # if (batch_idx >= orig_len_train_dataloader-5):
        #     continue
        batch = next(train_loader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        # x, y, w, types, mqq, mlv, MCWts, mH = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
        
        optimizer.zero_grad()
        if (MODEL_ARCH=="TRANSFORMER") and USE_DROPOUT:
            temp_hook_fn_attn = functools.partial(dropout_hook, p=0.1, v=0)
            temp_hook_fn_mlp = functools.partial(dropout_hook, p=0.1, v=0)
            outputs = model.run_with_hooks(x[...,:-2], types, fwd_hooks=[
                (f'blocks.{i}.hook_attn_out', temp_hook_fn_attn) for i in range(model.cfg.n_layers)
            ] + [
                (f'blocks.{i}.hook_mlp_out', temp_hook_fn_mlp) for i in range(model.cfg.n_layers)
            ]
            ).squeeze()
        elif (MODEL_ARCH=="TFLENS_DEEPSETS_RESIDUAL_VARIABLE") and USE_DROPOUT:
            temp_hook_fn_attn = functools.partial(dropout_hook, p=model_cfg['dropout_p'], v=0)
            temp_hook_fn_mlp = functools.partial(dropout_hook, p=model_cfg['dropout_p'], v=0)
            outputs = model.run_with_hooks(x[...,:-2], types, fwd_hooks=[
                # (f'blocks.{i}.hook_attn_out', temp_hook_fn_attn) for i in range(model.cfg.n_layers)
            ] + [
                (f'blocks.{i}.hook_mlp_out', temp_hook_fn_mlp) for i in range(model.cfg.n_layers)
            ]
            ).squeeze()
        else:
            # outputs, cache = run_with_cache_and_singleAttention(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
            # outputs, cache = run_with_cache_and_minAttention(model, x[...,:4+int(TAG_INFO_INPUT)], types, 0.5, detach=False)
            outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
            outputs = outputs.squeeze()
        # assert(False)
        # loss = criterion(outputs, x[...,-1], types, N_CTX-1, max_n_objs_to_read, w, config['wandb'], cache, x[...,:4])
        loss = criterion(cache, outputs, x[...,-1], types, N_CTX-1, max_n_objs_to_read, w, config['wandb'], x[...,:4])
        train_loss_epoch += loss.item() * w.sum().item()
        sum_weights_epoch += w.sum().item()
        
        loss.backward()
        optimizer.step()
        
        # Update training metrics
        total_train_samples_processed += len(y)
        train_metrics.update(outputs, x[...,-1], w, dsids, types)
        train_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)
        if (n_step % 10) == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, orig_len_train_dataloader, loss.item()))
        # Log training metrics every log_interval batches
        if ((batch_idx % log_interval) == (log_interval-1)):
            # assert(False)
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if config['wandb']:
                wandb.log({"train/lr": current_lr}, commit=False)
                wandb.log({"train_samps_processed": total_train_samples_processed}, commit=False)
            log_level = 2 if ((batch_idx % (longer_log_interval)) == (longer_log_interval-1)) else 0
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=log_level, save=config['wandb'], commit=False)
            train_metrics_MCWts.compute_and_log(epoch, prefix="train_MC", step=global_step, log_level=log_level, save=config['wandb'], commit=True)
            # train_metrics.reset_starts()
            # train_metrics_MCWts.reset_starts()
    
    # if config['wandb']:
    if 1:
        if config['wandb']:
            wandb.log({'train/loss_total':train_loss_epoch/sum_weights_epoch}, commit=False)
        if (epoch % 1) == 0:
            log_level = 3
            train_metrics.reset_starts()
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=log_level, save=config['wandb'], commit=False)
            train_metrics_MCWts.reset_starts()
            train_metrics_MCWts.compute_and_log(epoch, prefix="train_MC", step=global_step, log_level=log_level, save=config['wandb'], commit=True)
            # # Log learning rate
            # current_lr = optimizer.param_groups[0]['lr']
            # if config['wandb']:
            #     wandb.log({"train/lr": current_lr})#, step=global_step)
            if 0:
                # Log sample predictions
                sample_probs = F.softmax(outputs[:5], dim=1).detach().cpu().numpy()
                sample_preds = outputs[:5].argmax(dim=1).detach().cpu().numpy()
                sample_targets = y[:5].detach().cpu().numpy()
                if config['wandb']:
                    wandb.log({
                        "train/sample_predictions": wandb.Table(
                            columns=["Target", "Predicted", "Probabilities"],
                            data=[
                                [sample_targets[i], sample_preds[i], sample_probs[i]] 
                                for i in range(len(sample_targets))
                            ]
                        )
                    }, commit=False)
    
    if (epoch % 1) == 0:
        # Validation phase
        model.eval()
        with torch.no_grad():
            orig_len_val_dataloader=len(val_loader)
            loss = 0
            wt_sum = 0
            for batch_idx in range(orig_len_val_dataloader):
                batch = next(val_loader)
                # if (batch_idx >= orig_len_val_dataloader-5):
                #     continue

                x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                # x, y, w, types, mqq, mlv, MCWts = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
                
                # outputs, cache = run_with_cache_and_singleAttention(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
                # outputs, cache = run_with_cache_and_minAttention(model, x[...,:4+int(TAG_INFO_INPUT)], types, 0.5, detach=False)
                outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
                outputs = outputs.squeeze()
                
                # loss += criterion(outputs, x[...,-1], types, N_CTX-1, max_n_objs_to_read, w, config['wandb'], cache, x[...,:4]).sum() * w.sum()
                loss += criterion(cache, outputs, x[...,-1], types, N_CTX-1, max_n_objs_to_read, w, config['wandb'], x[...,:4]).sum() * w.sum()
                wt_sum += w.sum()
                val_metrics.update(outputs, x[...,-1], w, dsids, types)
                val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)
                # print('[%d/%d][%d/%d] Val' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader))
                if (n_step % 10) == 0:
                    print('[%d/%d][%d/%d]\tVAL Loss_C: %.4e' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader, loss.item()/wt_sum.item()))
            if config['wandb']:
                wandb.log({
                    "val/loss_total": loss.item(),
                    "val/loss_ce": loss.item()/wt_sum.item(),
                    # "loss/qq_mass": qq_mass_loss.item(),
                    # "loss/lv_mass": lv_mass_loss.item()
                }, 
                commit=False
                )
            # print('[%d/%d][%d/%d]\tVAL Loss_C: %.4e' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader, loss.item()/wt_sum.item()))
        
        # Log validation metrics
        if config['wandb']:
            log_level = 3
            val_metrics.compute_and_log(epoch, prefix="val", log_level=log_level, save=config['wandb'], commit=False)
            val_metrics_MCWts.compute_and_log(epoch, prefix="val_MC", log_level=log_level, save=config['wandb'], commit=True)

    
    # Log model gradients and parameters
    if 0:
        if (epoch % 5 == 0) and (config['wandb']):
            if 0:
                gradients = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients.append((name, wandb.Histogram(param.grad.cpu().numpy())))
                if config['wandb']:
                    wandb.log({"gradients": gradients}, commit=False)
            elif 0:
                gradients = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_values = param.grad.cpu().flatten().numpy() # Get the flattened gradient values and use them for the histogram
                        histogram_data = { # Create the histogram data as a dictionary
                            "values": grad_values,
                            "edges": [float(i) for i in range(len(grad_values) + 1)]  # create dummy edges for simplicity
                        }
                        gradients.append((name, histogram_data))
                if config['wandb']:
                    wandb.log({"gradients": gradients}, commit=True) # Log the gradients (values and edges) to WandB
            else:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({"gradients/%s"%(name ): wandb.Histogram(param.grad.detach().cpu().numpy())}, commit=False) # Log the gradients (values and edges) to WandB
            if 0:
                params = []
                for name, param in model.named_parameters():
                    params.append((name, wandb.Histogram(param.cpu().detach().numpy())))
                if config['wandb']:
                    wandb.log({"parameters": params})
            elif 0:
                params = []
                for name, param in model.named_parameters():
                    param_values = param.cpu().detach().numpy().flatten() # Get the parameter values as a flattened numpy array
                    params.append((name, param_values)) # Log the parameter values (no need for wandb.Histogram)
                if config['wandb']:
                    wandb.log({"parameters": params}) # Log the parameters
            else:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({"gradients/%s"%(name ): wandb.Histogram(param.detach().cpu().numpy())}, commit=False) # Log the gradients (values and edges) to WandB
    if ((epoch % SAVE_MODEL_EVERY) == (SAVE_MODEL_EVERY-1)):
        try:
            modelSaveDir = "%s/models/Nplits%d_ValIdx%d/"%(saveDir, n_splits, validation_split_idx)
            os.makedirs(modelSaveDir, exist_ok=True)
            torch.save(models[model_n]["model"].state_dict(), modelSaveDir + "/chkpt%d_%d" %(epoch, global_step) + '.pth')
        except:
            pass
# wandb.finish()

# %%
modelSaveDir = "%s/models/Nplits%d_ValIdx%d/"%(saveDir, n_splits, validation_split_idx)
os.makedirs(modelSaveDir, exist_ok=True)
torch.save(models[model_n]["model"].state_dict(), modelSaveDir + "/chkpt%d" %(global_step) + '.pth')

# %%
inclusion = torch.nn.functional.one_hot(((x[...,-1]==1) + (x[...,-1]>1)*2).to(int))
padding_token=N_CTX-1
categorical=True
print(types.shape)
print(inclusion.shape)
check_valid(types[:5], inclusion[:5], padding_token, categorical)
# %%

for i in model.named_parameters():
    print(f"{i[0]}: {sum([p.numel() for p in  i[1]])}")
# list(model.named_parameters())[0][1].numel()
