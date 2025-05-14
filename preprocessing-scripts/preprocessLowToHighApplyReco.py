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
from utils.utils import Get_PtEtaPhiM_fromXYZT, GetXYZT_FromPtEtaPhiM, GetXYZT_FromPtEtaPhiE, Rotate4VectorPhi, Rotate4VectorEta, Rotate4VectorPhiEta
from torch import nn
from utils.utils import Get_PtEtaPhiM_fromXYZT, deltaR
# import datasets

# %% Some basic setup
# Some choices about the  process
USE_ONE_NET = False
ONLY_CORRECT_RECO_FOR_TRAINING = True # Only include the correct reco, for training high-level classification networks with *just* lvbb or qqbb reco events
ONLY_CORRECT_TRUTH_FOR_TRAINING = True # Only include the correct truth, for training high-level classification networks with *just* lvbb or qqbb reco events
IS_CATEGORICAL = True
REMOVE_WHERE_TRUTH_WOULD_BE_CUT = False # Only want this if we are TRAINING the RECONSTRUCTION net! For predicting reco, or for training/predicting classification, we want these events to be present!
INCLUDE_ALL_SELECTIONS = True
INCLUDE_NEGATIVE_SELECTIONS = True
PHI_ROTATED = False
INCLUDE_TAG_INFO = True
TOSS_UNCERTAIN_TRUTH = True
USE_LORENTZ_INVARIANT_FEATURES = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = False
USE_MEDIUM_OLD_TRUTH_SETTING  = False
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here or not
assert(not (USE_OLD_TRUTH_SETTING and USE_MEDIUM_OLD_TRUTH_SETTING))
if (not INCLUDE_ALL_SELECTIONS) and INCLUDE_NEGATIVE_SELECTIONS:
    assert(False)
SHUFFLE_OBJECTS = False
CONVERT_TO_PT_PHI_ETA_M = False
MH_SEL = False
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
max_n_objs = 30 # BE CAREFUL because if we predict with more objects than we write/read, then we may not be able to reconstruct the same mass for the H/W/H+
max_n_objs_for_pred = 15
OUTPUT_DIR = '/data/atlas/baines/20250428v1_HighLevelAppliedRecoNNSplit_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_RemovedWrongRecoForTraining'*ONLY_CORRECT_RECO_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + 'semi_shuffled_'*SHUFFLE_OBJECTS + f'{max_n_objs}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
OUTPUT_DIR = '/data/atlas/baines/20250429v1_HighLevelAppliedRecoNNSplit/'
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

# %%
class RunningStats:
    def __init__(self, n_features):
        self.n = 0  # Count of data points seen
        self.mean = np.zeros(n_features)  # Mean for each feature
        self.m2 = np.zeros(n_features)  # Sum of squared differences for variance
    
    def update(self, x):
        # x is the batch of data, where each row is a data point, and columns are features
        n_batch = x.shape[0]
        
        # Update count
        self.n += n_batch
        if n_batch > 0:    
            # Update mean
            delta = x - self.mean
            self.mean += delta.sum(axis=0) / self.n
            
            # Update m2 (sum of squared differences)
            delta2 = x - self.mean
            self.m2 += (delta * delta2).sum(axis=0)
    
    def compute_std(self):
        # Calculate standard deviation using the variance (m2 / n - 1)
        return np.sqrt(self.m2 / (self.n - 1))

    def get_mean(self):
        return self.mean
    
    def get_variance(self):
        return self.m2 / (self.n - 1)  # Variance for each feature

# Initialize running stats for 7 features (based on your input features)
n_features = 7  # Adjust to match the number of input features you're tracking
running_stats = {
    'lvbb':RunningStats(n_features),
    'qqbb':RunningStats(n_features),
}

# %%
def last_true_index(arr):
    # Apply the condition to the array
    mask = arr!=0
    # Flip the array along the object axis
    flipped = np.flip(mask, axis=1)
    # Find the first True element in the flipped array
    first_true = np.argmax(flipped, axis=1)
    # Calculate the last true index
    last_true = arr.shape[1] - 1 - first_true
    return last_true


from utils.utils import check_valid


















































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
class DeepSetsWithResidualSelfAttentionVariableTrueSkip(nn.Module):
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

# %%
# Create a new model
if IS_CATEGORICAL:
    num_classes=3
else:
    num_classes=0
device = 'cuda'
model_cfg = {'d_model': 256, 'dropout_p': 0.2, "embedding_size":10, "num_heads":4}
DeepSetsWithResidualSelfAttentionVariableTrueSkip
if USE_ONE_NET:
    model = DeepSetsWithResidualSelfAttentionTriple(num_classes=num_classes, input_dim=N_Real_Vars-2, hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
else:
    models = {
        0:DeepSetsWithResidualSelfAttentionVariableTrueSkip(num_attention_blocks=3, include_mlp=False, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=200,  dropout_p=0.1,  num_heads=4, embedding_size=10).to(device),
        1:DeepSetsWithResidualSelfAttentionVariableTrueSkip(num_attention_blocks=3, include_mlp=False, num_classes=3, input_dim=N_Real_Vars-2, hidden_dim=200,  dropout_p=0.1,  num_heads=4, embedding_size=10).to(device),
    }

if 1: # 
    print("WARNING: You are starting from a semi-pre-trained model state")
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250302-224613_TrainingOutput/models/0/chkpt74_414975.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250304-145036_TrainingOutput/models/0/chkpt163050.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250305-195515_TrainingOutput/models/0/chkpt163050.pth" # DSSAR d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250305-232917_TrainingOutput/models/0/chkpt162930.pth" # 
    if USE_ONE_NET:
        modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250307-191542_TrainingOutput/models/0/chkpt162930.pth" # DSSAR3 d_model=256, n_heads=4, num_embedding=10,
        loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
        model.load_state_dict(loaded_state_dict)
        modeltag = modelfile.split('/')[-4].split('_')[0]
    else:
        modelfile1="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250321-124811_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_54830.pth"
        modelfile1="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250321-124811_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth"
        loaded_state_dict = torch.load(modelfile1, map_location=torch.device(device))
        models[0].load_state_dict(loaded_state_dict)

        modelfile2="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250321-124953_TrainingOutput/models/Nplits2_ValIdx1/chkpt9_54880.pth"
        modelfile2="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250321-124953_TrainingOutput/models/Nplits2_ValIdx1/chkpt29_164640.pth"
        loaded_state_dict = torch.load(modelfile2, map_location=torch.device(device))
        models[1].load_state_dict(loaded_state_dict)
        modeltag = modelfile1.split('/')[-4].split('_')[0] + modelfile2.split('/')[-4].split('_')[0]
        












































# %%
# Main function to actually process a single file and return the arrays
def process_single_file(filepath, max_n_objs, shuffle_objs, target_channel):
    assert((target_channel == 'lvbb') or (target_channel == 'qqbb'))
    if USE_OLD_TRUTH_SETTING:
        truth_var = 'truth_W_decay_mode'
        mH_var = 'mH'
    elif USE_MEDIUM_OLD_TRUTH_SETTING:
        truth_var = 'll_truth_decay_mode_old'
        mH_var = 'll_best_mH'
    else:
        truth_var = 'll_truth_decay_mode'
        mH_var = 'll_best_mH'
    particle_features=['part_px', 'part_py', 'part_pz', 'part_energy']
    if INCLUDE_TAG_INFO:
        particle_features.append('ll_particle_tagInfo')
    if INCLUDE_INCLUSION_TAGS: # this means we're expecting the reco/true inclusion tags for each particle too
        particle_features += ['ll_particle_recoInclusion','ll_particle_trueInclusion']
    particle_features.append('ll_particle_type')
    x_part, x_event, y = read_file(filepath, 
                                max_num_particles=100, # This should be high enough that you never end up cutting off an object which is present in the truth matching. In reality, this just needs to be 25 for the samples I have
                                particle_features=particle_features,
                                event_level_features=[
                                    'eventWeight',
                                    'selection_category',
                                    'MET',
                                    'll_best_mWH_qqbb',
                                    'll_best_mWH_lvbb',
                                    mH_var,
                                    'll_successful_truth_match',
                                    'eventNumber',
                                ],
                                labels=['DSID', truth_var, 'eventNumber'],
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
    mWh_qqbb = x_event[:,3]
    mWh_lvbb = x_event[:,4]
    mH = x_event[:,5]
    successful_truth_match = x_event[:,6]
    eventNumber = y[:,2]
    if USE_OLD_TRUTH_SETTING:
        mH = mH*1e3
    if MET_CUT_ON:
        met = x_event[:,2]

    # Remove events with no large-R jets and which don't pass selection category and which have too low Met Pt
    if INCLUDE_ALL_SELECTIONS:
        if INCLUDE_NEGATIVE_SELECTIONS:
            selection_removals = np.zeros_like(selection_category<0)
        else:
            selection_removals = selection_category<0
    else:
        selection_removals = ~((selection_category == 0) | (selection_category == 3) | (selection_category == 8) | (selection_category == 9) | (selection_category == 10))
    is_sig = ((y[:,0] > 500000) & (y[:,0] < 600000))
    # Remove events where the truth match is not successful
    successful_truth_match_removals = ~(successful_truth_match.astype(bool)) & is_sig
    # Get rid of events where we don't keep enough objects to keep all the relevant truth objects
    true_inclusion = x_part[:, -1, :]
    if REMOVE_WHERE_TRUTH_WOULD_BE_CUT:
        keeping_all_truth_removal = (last_true_index(true_inclusion!=0)>max_n_objs-1) & is_sig
    else:
        keeping_all_truth_removal = np.zeros_like(selection_category==0)
    if not REQUIRE_XBB:
        no_ljet = (((type_part==3)|(type_part==5)).sum(axis=1) == 0)
    else:
        no_ljet = ((type_part==5).sum(axis=1) == 0)
    if MET_CUT_ON:
        low_MET = met < 30e3 # 30GeV Minimum for MET
    else:
        low_MET = np.zeros_like(no_ljet)#.astype(bool)
    if TOSS_UNCERTAIN_TRUTH:
        uncertain_cut = (((y[:, 1] != 1) & (y[:, 1] != 2)) & is_sig)
    else:
        uncertain_cut = np.zeros_like(no_ljet)#.astype(bool)
    
    num_samps = x_part.shape[0]
    outputs = np.zeros((num_samps, x_part.shape[2], 3))
    # pred_inclusion = np.zeros(num_samps, max_n_objs)
    valid = np.zeros(num_samps)
    predicted_channel = np.zeros(num_samps)
    if (num_samps>0):
        batch_size = 256
        num_batches = (num_samps-1)//batch_size + 1
        with torch.no_grad():
            for batch_idx in range(num_batches):
                if USE_ONE_NET:
                    outputs_batch = model(torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                else:
                    outputs_batch0 = models[0](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                    outputs_batch1 = models[1](torch.Tensor(einops.rearrange(x_part[batch_idx*batch_size:(batch_idx+1)*batch_size, 1:5+1, :max_n_objs_for_pred], 'b v o-> b o v')).to(device)*torch.tensor([[[1e-5, 1e-5, 1e-5, 1e-5, 1]]]).to(device), torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred]).to(device).to(int)).squeeze()
                    # Now get the relevant prediction (from the net it was val set in, NOT the net it was train set in, based on eventNumber)
                    outputs_batch = outputs_batch0 * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,])%2)==0).to(float).to(device).reshape(-1,1,1) + outputs_batch1 * ((torch.Tensor(eventNumber[batch_idx*batch_size:(batch_idx+1)*batch_size,])%2)==1).to(float).to(device).reshape(-1,1,1)

                outputs[batch_idx*batch_size:(batch_idx+1)*batch_size, :max_n_objs_for_pred] = outputs_batch.cpu().numpy()
                # pred_inclusion[batch_idx*batch_size:(batch_idx+1)*batch_size] = (outputs.argmax(dim=-1)>0).cpu().numpy()
                valid_batch, predicted_channel_batch = check_valid(torch.Tensor(type_part[batch_idx*batch_size:(batch_idx+1)*batch_size,:max_n_objs_for_pred]), outputs_batch.cpu(), N_CTX-1, IS_CATEGORICAL, returnTypes=True)
                valid[batch_idx*batch_size:(batch_idx+1)*batch_size] = valid_batch.cpu().numpy()
                predicted_channel[batch_idx*batch_size:(batch_idx+1)*batch_size] = predicted_channel_batch.cpu().numpy()
                # print(f"Processed batch {batch_idx}/{num_batches}")
        not_valid = ~(valid.astype(bool))
        print(f"{y[0,0]}: {not_valid.sum():7d}/{len(not_valid):7d} ({not_valid.sum()/len(not_valid)*100:5.2f})")
        print(f"{y[0,0]}: {(not_valid*event_weights).sum():10.2f}/{event_weights.sum():10.2f} ({(not_valid*event_weights).sum()/event_weights.sum()*100:5.2f})")
    else:
        print("Skipping since num_samps==0")
        not_valid=np.zeros(num_samps).astype(bool)

    # Get H+ mass
    fourmom=(einops.rearrange(x_part, 'b v o-> b o v')[:, :max_n_objs_for_pred, 1:4+1] * einops.rearrange((outputs.argmax(axis=-1)>0)[:, :max_n_objs_for_pred], 'b o -> b o 1'))
    _,_,_,mWh=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(axis=-1),fourmom[...,1].sum(axis=-1),fourmom[...,2].sum(axis=-1),fourmom[...,3].sum(axis=-1))
    mWh_lvbb = mWh * (predicted_channel==1)
    mWh_qqbb = mWh * (predicted_channel==2)
    
    # Get H mass
    fourmom=(einops.rearrange(x_part, 'b v o-> b o v')[:, :max_n_objs_for_pred, 1:4+1] * einops.rearrange((outputs.argmax(axis=-1)==1)[:, :max_n_objs_for_pred], 'b o -> b o 1'))
    _,_,_,mH=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(axis=-1),fourmom[...,1].sum(axis=-1),fourmom[...,2].sum(axis=-1),fourmom[...,3].sum(axis=-1))

    if MH_SEL:
        mH_cut = (mH < 95e3) | (mH > 140e3)
    else:
        mH_cut = np.zeros_like(no_ljet)#.astype(bool)


    
    if len(y):
        dsid = y[0,0]
    else:
        dsid = 0 # Let the function run with empty arrays anyway
    if target_channel == 'lvbb':
        channel_num = 1
    elif target_channel == 'qqbb':
        channel_num = 2

    if ONLY_CORRECT_TRUTH_FOR_TRAINING:
        truth_removal = (y[:,1]!=channel_num) if ((dsid>500000) and (dsid<600000)) else np.zeros_like(y[:,1]).astype(bool)
    else:
        truth_removal = np.zeros_like(y[:,1]).astype(bool)
    if ONLY_CORRECT_RECO_FOR_TRAINING:
        wrong_reco_removal = (predicted_channel!=channel_num)
    else:
        wrong_reco_removal = np.zeros_like(y[:,1]).astype(bool)

    combined_removal = (truth_removal | wrong_reco_removal | not_valid | no_ljet | selection_removals | low_MET | mH_cut | uncertain_cut | keeping_all_truth_removal | successful_truth_match_removals)
    removals = {0: (combined_removal & (~is_sig)).sum(),
                1: (combined_removal & is_sig).sum()}
    x_part = x_part[~combined_removal]
    y = y[~combined_removal]
    event_weights = event_weights[~combined_removal]
    mWh_qqbb = mWh_qqbb[~combined_removal]
    mWh_lvbb = mWh_lvbb[~combined_removal]
    mWh = mWh[~combined_removal]
    mH = mH[~combined_removal]
    type_part = type_part[~combined_removal]
    pred_inclusion = outputs.argmax(axis=-1)[~combined_removal]
    eventNumber = eventNumber[~combined_removal]
    

    # HERE Need to write some code to store this properly
    # dsid = np.searchsorted(dsid_set, y[:, 0])
    dsid = y[:, 0]
    truth_label = (y[:, 1] == 1)*1 + (y[:, 1] == 2)*2
    # truth_label = np.searchsorted(types_set, y[:, 1])

    

    # Reshape the x array to how we want to read it in later
    x_part = einops.rearrange(x_part, 'batch d_input object -> batch object d_input')



    # Here, we calculate the relevant variables for high-level processing
    # The Lepton (L) is made of all particles with x_part[...,-1] == 0 or 1
    # The Higgs (H) is made of all particles with pred_inculsion == 1
    # The W boson (W) is made of all particles with pred_inculsion == 2
    lepton_mask = ((x_part[...,0]==0)|(x_part[...,0]==1)).astype(int)
    lep_fourvec = (x_part * np.expand_dims(lepton_mask, axis=-1))[..., 1:4+1].sum(axis=1)
    higgs_mask = (pred_inclusion==1).astype(int)
    higgs_fourvec = (x_part * np.expand_dims(higgs_mask, axis=-1))[..., 1:4+1].sum(axis=1)
    w_mask = (pred_inclusion==2).astype(int)
    w_fourvec = (x_part * np.expand_dims(w_mask, axis=-1))[..., 1:4+1].sum(axis=1)
    w_lep_mask = ((x_part[...,0]==0)|(x_part[...,0]==1)|(x_part[...,0]==2)).astype(int)
    w_lep_fourvec = (x_part * np.expand_dims(w_lep_mask, axis=-1))[..., 1:4+1].sum(axis=1)
    if target_channel == 'lvbb':
        deltaR_LH, _, _ = deltaR(lep_fourvec, higgs_fourvec)
        _, deltaPhi_HWlep, deltaEta_HWlep = deltaR(higgs_fourvec, w_fourvec)
        ratio_Hpt_mVH_lvbb = (np.sqrt(higgs_fourvec[:,0]**2 + higgs_fourvec[:,1]**2) / mWh_lvbb)
        ratio_Wpt_mVH_lvbb = (np.sqrt(w_fourvec[:,0]**2 + w_fourvec[:,1]**2) / mWh_lvbb)
        mTop_lepto = -1*np.ones_like(ratio_Wpt_mVH_lvbb) # Default in case we can't reconstruct the top leptonically
        # Get top mass from Wlep and btagged small-R jet. The jet must pass a btag threshold, and be a sufficient distance outside the Higgs
        # First, check it's a small jet
        quark_inclusion_in_top = (x_part[...,0]==4) # Shape [batch object]
        # Now check angular distance to Higgs
        _, deltaPhi_H_all, deltaEta_H_all = deltaR(np.expand_dims(higgs_fourvec, 1), x_part[...,1:1+4]) # Each has shape [batch object]
        quark_inclusion_in_top = quark_inclusion_in_top & ((np.sqrt((deltaPhi_H_all)**2 + (deltaEta_H_all)**2) > 1.4))
        # THESHOLD_FOR_BTAG = 0.645925
        # quark_inclusion_in_top = quark_inclusion_in_top & (x_part[...,5] > THESHOLD_FOR_BTAG)
        # Now calculate the mass of the top created by combining Wlep with each of these possible quarks. We'll just calculate the mass with each of the objects, and then mask for acceptable quarks later.
        top_masses = Get_PtEtaPhiM_fromXYZT(x_part[...,1] + np.expand_dims(w_lep_fourvec[:,0], 1), x_part[...,2] + np.expand_dims(w_lep_fourvec[:,1], 1), x_part[...,3] + np.expand_dims(w_lep_fourvec[:,2], 1), x_part[...,4] + np.expand_dims(w_lep_fourvec[:,3], 1))[3] # Shape [batch object]
        # Now, get the one which minimises abs(true_top_mass - top_mass)/(0.15*top_mass) AND uses one of the allowed quarks
        # We need to do this in a way that doesn't require a for loop, so we need to use broadcasting
        # Get the top mass from the truth
        true_top_mass = 172.76e3
        top_mass_resolution = abs(true_top_mass-top_masses)/(0.15*top_masses) # Shape [batch object]
        # Now, we need to get the minimum of this, but only for the quarks which are allowed
        top_mass_resolution[~quark_inclusion_in_top] = np.inf # Set the top mass resolution to be infinite for the quarks which are not allowed
        best_quark_idx = np.argmin(top_mass_resolution, axis=1) # Shape [batch]
        best_quark_idx[quark_inclusion_in_top.sum(axis=1)==0] = 0 # If there are no quarks, set the best quark index to be 0 (which is the lepton/neutrino)
        mTop_lepto = np.where(best_quark_idx==0, -1, top_masses[np.arange(top_masses.shape[0]), best_quark_idx]) # Shape [batch]
        
        # The LepEnergyFracLep is the ratio of Wlep Pt divided by sum of (Wlep Pt + Higgs Pt)
        LepEnergyFracLep = np.sqrt(w_lep_fourvec[:,0]**2 + w_lep_fourvec[:,1]**2) / (np.sqrt(w_lep_fourvec[:,0]**2 + w_lep_fourvec[:,1]**2) + np.sqrt(higgs_fourvec[:,0]**2 + higgs_fourvec[:,1]**2))
        high_level_vars = np.stack([deltaR_LH,np.abs(deltaPhi_HWlep),np.abs(deltaEta_HWlep),ratio_Hpt_mVH_lvbb,ratio_Wpt_mVH_lvbb,mTop_lepto,LepEnergyFracLep], axis=-1) # Shape [batch 7]
        # high_level_vars = np.stack([deltaR_LH,deltaPhi_HWlep,deltaEta_HWlep,detlaPhi_LH,deltaEta_LH,mTop_lepto,LepEnergyFracLep], axis=-1) # Shape [batch 7]
    elif target_channel == 'qqbb':
        deltaR_LH, _, _ = deltaR(lep_fourvec, higgs_fourvec) # Shape [batch]
        _, deltaPhi_HWhad, deltaEta_HWhad = deltaR(higgs_fourvec, w_fourvec)
        ratio_Hpt_mVH_qqbb = (np.sqrt(higgs_fourvec[:,0]**2 + higgs_fourvec[:,1]**2) / mWh_qqbb)
        ratio_Wpt_mVH_qqbb = (np.sqrt(w_fourvec[:,0]**2 + w_fourvec[:,1]**2) / mWh_qqbb)
        deltaR_LWhad, _, _ = deltaR(w_fourvec, lep_fourvec)
        # LepEnergyFracHad is the ratio of Wlep Pt divided by sum of (Wlep Pt + WHad Pt + Higgs Pt)
        LepEnergyFracHad = np.sqrt(w_lep_fourvec[:,0]**2 + w_lep_fourvec[:,1]**2) / (np.sqrt(w_lep_fourvec[:,0]**2 + w_lep_fourvec[:,1]**2) + np.sqrt(higgs_fourvec[:,0]**2 + higgs_fourvec[:,1]**2) + np.sqrt(w_fourvec[:,0]**2 + w_fourvec[:,1]**2))
        high_level_vars = np.stack([deltaR_LH,np.abs(deltaPhi_HWhad),np.abs(deltaEta_HWhad),ratio_Hpt_mVH_qqbb,ratio_Wpt_mVH_qqbb,deltaR_LWhad,LepEnergyFracHad], axis=-1) # Shape [batch 7]
    else:
        assert(False)

    x_part = np.concatenate(
        [
            x_part[...,:-1],
            einops.rearrange(pred_inclusion,'b o -> b o 1'),
            einops.rearrange(x_part[...,-1],'b o -> b o 1'),
        ],
        axis=-1
    )
    if shuffle_objs: # # Shuffle the tensors along the objects dimension (keeping the mapping bettween X_train objects and types_train objects the same)
        # non_zero_mask = x[:, :, 0] != 0  # Mask for non-zero elements in the first variable
        # permutation_indices = torch.argsort(torch.rand(*x_part[:,:,0].shape), dim=-1)
        # batch_indices = torch.arange(x_part.shape[0]).unsqueeze(1).expand(x_part.shape[0], x_part.shape[1])
        # x_part = x_part[batch_indices, permutation_indices]
        num_non_zero = (x_part[:,:,0]!=N_CTX-1).sum(axis=1)
        result = x_part.copy()
        if 1: # Old style - shuffle only the non-zero ones
            for i in range(x_part.shape[0]):  # For each batch element
                n_non_zero = num_non_zero[i]
                if n_non_zero > 1:
                    # Generate permutation indices for the non-zero objects in this batch
                    permuted_indices = np.random.permutation(n_non_zero)
                    # Shuffle the non-zero elements along the object dimension
                    result[i,:n_non_zero] = result[i, permuted_indices]
        else: # Shuffle all up to max_n objs
            for i in range(x_part.shape[0]):  # For each batch element
                n_non_zero = num_non_zero[i]
                if n_non_zero > 1:
                    # Generate permutation indices for the non-zero objects in this batch
                    permuted_indices = np.random.permutation(max_n_objs)
                    # Shuffle the non-zero elements along the object dimension
                    result[i,:max_n_objs] = result[i, permuted_indices]
        return result[:,:max_n_objs,:N_Real_Vars+2], truth_label, dsid, event_weights, removals, mWh_qqbb, mWh_lvbb, mH, eventNumber
    else:
        return high_level_vars, truth_label, dsid, event_weights, removals, mWh, mH, eventNumber

def combine_arrays_for_writing(x_chunk, y_chunk, dsid_chunk, weights_chunk, mWh_chunk, mH_chunk, eventNumber_chunk):

    array_to_write=np.float32(np.concatenate(
        [y_chunk.reshape(-1,1), 
         weights_chunk.reshape(-1,1), 
         mWh_chunk.reshape(-1,1), 
         dsid_chunk.reshape(-1,1), 
         mH_chunk.reshape(-1,1), 
         (eventNumber_chunk%1000).reshape(-1,1), # Have to make this remainder some small-ish number, otherwise the numbers might be too big for the np.float32 and will come out as zero. Currently plan to just use for train/val split, so definitely shouldn't need more than eg. 1000
         x_chunk,
         ],
    axis=-1
    ))
    return array_to_write


# %%
types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'Xbb_ljet'}
# DATA_PATH='/data/atlas/HplusWh/20241218_SeparateLargeRJets_NominalWeights/'
# DATA_PATH='/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/'
# DATA_PATH='/data/atlas/HplusWh/20250218_Cats038910_NoDeltaRReq_TagInfo/'
DATA_PATH='/data/atlas/HplusWh/20250227_v4_tmpWithTrueInclusion/' # For background this is okay
# DATA_PATH='/data/atlas/HplusWh/20250305_WithTrueInclusion_FixedOverlapWHsjet/' # For signal must be this!
DATA_PATH='/data/atlas/HplusWh/20250313_WithTrueInclusion_FixedOverlapWHsjet_SmallJetCloseToLargeJetRemovalDeltaR0.5/'
MAX_CHUNK_SIZE = 100000
# MAX_PER_DSID = {dsid : 10000000 for dsid in dsid_set}
# MAX_PER_DSID[410470] = 100
for dsid in dsid_set:
    for channel in ['lvbb', 'qqbb']:
        # if dsid <= 700341:
        # if dsid != 410470:
        # if dsid > 363356:
        # if dsid != 410470:
        # if dsid != 510120:
            # continue
        # if '510' in str(dsid):
        #     continue
        # if (dsid < 500000) or (dsid > 600000): # Is background
        # if (dsid < 500000) or (dsid > 600000): # Is background
        # if (dsid > 500000) and (dsid < 600000): # Is signal
        #     continue
        # if dsid < 500000:
        #     continue
    # for dsid in [510120]:
        # if ((500000<dsid) and (600000>dsid)) or (dsid==410470):
        # if (dsid==410470):
        # if (dsid==700349):
        if True:
        # if ((500000<dsid) and (600000>dsid)):
        # if (dsid==410470):
        # # if (dsid==510120):
        # if dsid > 700338:
        # if (dsid ==510121):
        # if ((dsid < 600000) and (dsid > 500000)) or (dsid==410470) or (dsid==407342):
            pass
        else:
            continue
        nfs = 0
        removals = {0:0, 1:0}
        # if ((dsid > 500000) and (dsid < 600000)):# or (dsid==410470):
        # if (dsid==410470):
        #     pass
        # else:
        #     continue
        all_files = []
        for filename in os.listdir(DATA_PATH):
            if (str(dsid) in filename) and (filename.endswith('.root') and (filename.startswith('user'))):
                all_files.append(DATA_PATH + '/' + filename)
        # if dsid != 510122:
        #     continue
        x_parts=[]
        ys=[]
        dsids=[]
        weights = []
        mWhs = []
        mHs = []
        eventNumbers = []
        current_chunk_size = 0
        total_events_written_for_sample = 0
        total_entries_written_for_sample = 0
        sum_abs_weights_written_for_sample = 0
        sum_weights_written_for_sample = 0
        # if (dsid > 500000) and (dsid<600000) and (DATA_PATH!='/data/atlas/HplusWh/20250305_WithTrueInclusion_FixedOverlapWHsjet/'):
        #     assert(False)

        memmap_path = os.path.join(OUTPUT_DIR, f'dsid_{dsid}_{channel}.memmap')
        if len(all_files) > 0: # Safeguard for when there aren't any files to loop through, so we don't create an empty memmap file
            with open(memmap_path, 'wb') as f:
                pass  # Create empty file
        for file_n, path in enumerate(all_files):
            # print(path)
            # if path == '/data/atlas/HplusWh/20241128_ProcessedLightNtuples/user.rhulsken.mc16_13TeV.363355.She221_ZqqZvv.TOPQ1.e5525s3126r10201p4512.Nominal_v0_1l_out_root/user.rhulsken.31944615._000001.out.root':
            #     continue
            x_chunk, y_chunk, dsid_chunk, weights_chunk, removals_chunk, mWhs_chunk, mH_chunk, eventNumber_chunk = process_single_file(filepath=path, max_n_objs=max_n_objs, shuffle_objs=SHUFFLE_OBJECTS, target_channel=channel)
            x_parts.append(x_chunk)
            ys.append(y_chunk)
            dsids.append(dsid_chunk)
            weights.append(weights_chunk)
            mWhs.append(mWhs_chunk)
            mHs.append(mH_chunk)
            eventNumbers.append(eventNumber_chunk)
            # if x_chunk.shape[0]:
            #     assert(False)
            running_stats[channel].update(x_chunk)
            current_chunk_size += x_chunk.shape[0]
            if (current_chunk_size > MAX_CHUNK_SIZE) or (file_n+1 == len(all_files)):
                array_chunk = combine_arrays_for_writing(x_chunk=np.concatenate(x_parts, axis=0), y_chunk=np.concatenate(ys, axis=0), dsid_chunk=np.concatenate(dsids, axis=0), weights_chunk=np.concatenate(weights, axis=0), mWh_chunk=np.concatenate(mWhs, axis=0), mH_chunk=np.concatenate(mHs, axis=0), eventNumber_chunk=np.concatenate(eventNumbers, axis=0))
                # assert(False)
                if array_chunk.shape[0] > 0: # Check there's actually something in there
                    # memmap = np.memmap(memmap_path, dtype=BIN_WRITE_TYPE, mode='r+', offset=total_entries_written_for_sample*array_chunk.itemsize, shape=array_chunk.shape)
                    memmap = np.memmap(memmap_path, dtype=BIN_WRITE_TYPE, mode='r+', offset=total_entries_written_for_sample*array_chunk.itemsize, shape=array_chunk.shape)
                    memmap[:] = array_chunk[:]
                total_events_written_for_sample += array_chunk.shape[0]
                total_entries_written_for_sample += np.prod(array_chunk.shape)
                sum_abs_weights_written_for_sample += np.abs(np.concatenate(weights, axis=0)).sum()
                sum_weights_written_for_sample += np.concatenate(weights, axis=0).sum()
                # Save total number of events
                with open(memmap_path + '.shape', 'w') as f:
                    f.write(f"{total_events_written_for_sample},{sum_abs_weights_written_for_sample},{sum_weights_written_for_sample}")
                current_chunk_size = 0
                x_parts=[]
                ys=[]
                dsids=[]
                weights = []
                mWhs = []
                mHs = []
                eventNumbers = []
            removals[0]+=removals_chunk[0]
            removals[1]+=removals_chunk[1]
            nfs+=1
            if (nfs%10)==0:
                print("Processed %d files, have %d events in buffer, %d written so far" %(nfs, current_chunk_size, total_events_written_for_sample))
            if nfs > 1000000:
                break
            # if (total_events_written_for_sample > MAX_PER_DSID[dsid]):
            #     print("Reached %d for %s" %(total_events_written_for_sample, filename))
            #     break
        with open(memmap_path + '.shape', 'w') as f:
            f.write(f"{total_events_written_for_sample},{sum_abs_weights_written_for_sample},{sum_weights_written_for_sample}")
        print("Total accepted %s: %d" %(channel, total_events_written_for_sample))
        print("Total bkg removed for no ljet/MET Cut/Selection category fail: %d" %(removals[0]))
        print("Total sig removed for no ljet/MET Cut/Selection category fail: %d" %(removals[1]))

for channel in ['lvbb', 'qqbb']:
    print("Feature means: ", running_stats[channel].get_mean())
    print("Feature std devs: ", running_stats[channel].compute_std())
    # Optionally, you can save these stats for later use in a scaler
    np.save(os.path.join(OUTPUT_DIR, f'{channel}_mean.npy'), running_stats[channel].get_mean())
    np.save(os.path.join(OUTPUT_DIR, f'{channel}_std.npy'), running_stats[channel].compute_std())


# %%
if channel == 'lvbb':
    ns = ['deltaR_LH','deltaPhi_HWlep','deltaEta_HWlep','ratio_Hpt_mVH_lvbb','ratio_Wpt_mVH_lvbb','mTop_lepto','LepEnergyFracLep']
elif channel == 'qqbb':
    ns = ['deltaR_LH','deltaPhi_HWhad','deltaEta_HWhad','ratio_Hpt_mVH_qqbb','ratio_Wpt_mVH_qqbb','deltaR_LWhad','LepEnergyFracHad']
else:
    assert(False)
xc=np.concatenate(x_parts, axis=0)
plt.figure(figsize=(9,9))
for i in range(xc.shape[1]):
    plt.subplot(3,3,i+1)
    plt.title(ns[i])
    if ns[i]!='mTop_lepto':
        bins=20
        plt.hist(xc[:,i], weights=np.ones(len(xc))*1/(len(xc)), bins=bins)
    else:
        bins= np.arange(41)/40*800
        plt.hist(xc[xc[:,i]!=-1,i]*1e-3, weights=np.ones(len(xc[xc[:,i]!=-1]))*1/(len(xc[xc[:,i]!=-1,i])), bins=bins)
        plt.ylim([0, 0.28])
plt.tight_layout()
plt.show()
plt.close('all')
# %%
