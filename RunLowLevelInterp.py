# %% # Load required modules
# Import common modules
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from torch import nn
import shutil
from metrics.lowlevelrecometrics import HEPMetrics, HEPLossWithEntropy
import importlib
import json
import einops

# Import model classes from models.py
from models.models import TestNetwork

# Import utility functions
from utils.utils import check_valid, check_category, DSID_MASS_MAPPING, deltaR, Get_PtEtaPhiM_fromXYZT, print_vars, print_inclusion

# Import interp modules
from interp.activations import ModelActivationExtractor, ActivationCache, get_residual_stream, hook_attention_heads
from interp.attention import AttentionAnalyzer
from interp.causal_analysis import DirectLogitContributionAnalyzer, DirectLogitAttributor, ActivationPatcher
from interp.residual_analysis import flatten_selected_objects, fit_linear_probe, probe_accuracy, plot_pca
from interp.ablation import ablate_attention_head

# %% Configuration settings
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")

#########
# Data configuration settings
#########
ONLY_SIG = True  # Only look at signal samples
IS_CATEGORICAL = True
PHI_ROTATED = False
REMOVE_WHERE_TRUTH_WOULD_BE_CUT = True
TOSS_UNCERTAIN_TRUTH = True
USE_OLD_TRUTH_SETTING = False
SHUFFLE_OBJECTS = False
NORMALISE_DATA = False
SCALE_DATA = True
assert(not (NORMALISE_DATA and SCALE_DATA))
CONVERT_TO_PT_PHI_ETA_M = False
IS_XBB_TAGGED = False
REQUIRE_XBB = False
assert (~((REQUIRE_XBB and (~IS_XBB_TAGGED))))
INCLUDE_TAG_INFO = True
INCLUDE_ALL_SELECTIONS = True
INCLUDE_NEGATIVE_SELECTIONS = True
if (INCLUDE_NEGATIVE_SELECTIONS and (not INCLUDE_ALL_SELECTIONS)):
    assert(False)
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 3  # Number of target classes (needed for one-hot encoding)

#########
# Model configuration settings
#########
MODEL_ARCH = "DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP"
device = "cpu"  # 'mps' if torch.backends.mps.is_available() else 'cpu'
USE_DROPOUT = True
HAS_MLP = False  # Default, will be overwritten when loading model
EXCLUDE_TAG = False  # Default, will be overwritten when loading model
USE_LORENTZ_INVARIANT_FEATURES = True

#########
# Object settings
#########
if IS_XBB_TAGGED:
    N_CTX = 7  # 6 object types + padding
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
else:
    N_CTX = 6  # 5 object types + padding
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet'}
padding_token = N_CTX-1

#########
# Data processing settings
#########
BIN_WRITE_TYPE = np.float32
max_n_objs_in_file = 15
max_n_objs_to_read = 15
assert(max_n_objs_in_file == max_n_objs_to_read)
assert(REMOVE_WHERE_TRUTH_WOULD_BE_CUT)

if INCLUDE_TAG_INFO:
    N_Real_Vars = 5  # px, py, pz, energy, tagInfo
    N_Real_Vars_InFile = 7
else:
    N_Real_Vars = 4  # px, py, pz, energy
    N_Real_Vars_InFile = 6  # Plus naive-reco-inclusion, true-inclusion

# %% Data path configuration
batch_size = 3000

DATA_PATH = f'/data/atlas/baines/20250321v1_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + \
           '_NotPhiRotated'*(not PHI_ROTATED) + \
           '_XbbTagged'*IS_XBB_TAGGED + \
           f'_WithRecoMasses_{max_n_objs_in_file}' + \
           '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + \
           '_MetCut'*MET_CUT_ON + \
           '_XbbRequired'*REQUIRE_XBB + \
           '_mHSel'*MH_SEL + \
           '_OldTruth'*USE_OLD_TRUTH_SETTING + \
           '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH + \
           '_WithTagInfo'*INCLUDE_TAG_INFO + \
           '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS + \
           'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + \
           '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT + '/'

if 'AppliedRecoNN' in DATA_PATH:
    N_Real_Vars_InFile += 1  # Plus NN-reco-inclusion INBETWEEN THE naive-reco-inclusion and true-inclusion

# Data normalization settings
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

# %% Setup data loader
memmap_paths_train = {}
memmap_paths_val = {}
for file_name in os.listdir(DATA_PATH):
    if ('shape' in file_name) or ('npy' in file_name):
        continue
    dsid = file_name[5:11]
    if (int(dsid) > 500000) and (int(dsid) < 600000):
        memmap_paths_train[int(dsid)] = DATA_PATH + file_name
    else:
        pass  # Skip because we don't train the reco on bkg
    
    if ((int(dsid) > 500000) and (int(dsid) < 600000)) or (not (ONLY_SIG)):
        memmap_paths_val[int(dsid)] = DATA_PATH + file_name
    else:
        print("For now, don't include the background even in the val set")

n_splits = 2
validation_split_idx = 0
val_dataloader = ProportionalMemoryMappedDataset(
    memmap_paths=memmap_paths_val,  # DSID to memmap path
    max_objs_in_memmap=max_n_objs_in_file,
    N_Real_Vars_In_File=N_Real_Vars_InFile,
    N_Real_Vars_To_Return=N_Real_Vars_InFile,
    class_proportions=None,
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
print(f"Total samples in validation dataset: {val_dataloader.get_total_samples()}")

# %% Load model
print("Loading pre-trained model...")
if 0:
    # Test with low dimensionality network
    modelfile = "/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250429-104046_TrainingOutput/models/Nplits2_ValIdx0/chkpt19_109660.pth"
    HAS_MLP = False
    model = TestNetwork(
        hidden_dim_attn=2, 
        hidden_dim=100, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=None, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=4,
        hidden_dim_mlp=1, 
        num_heads=2, 
        embedding_size=10, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX
    ).to(device)
elif 0:
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250413-234444_TrainingOutput/models/Nplits2_ValIdx0/chkpt11_65796.pth"
    EXCLUDE_TAG = True
    HAS_MLP=False
    model = TestNetwork(
        hidden_dim_attn=None, 
        hidden_dim=300, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=1, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=3,
        hidden_dim_mlp=1, 
        num_heads=4, 
        embedding_size=10, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX,
        num_object_net_layers=1,
        is_layer_norm=True
    ).to(device)
elif 0:
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250328-113053_TrainingOutput/models/Nplits2_ValIdx0/chkpt24_137075.pth"
    EXCLUDE_TAG = False
    HAS_MLP=True
    model = TestNetwork(
        hidden_dim_attn=None, 
        hidden_dim=152, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=None, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=3,
        hidden_dim_mlp=400, 
        num_heads=4,
        embedding_size=10, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX,
        num_object_net_layers=1,
        is_layer_norm=True
    ).to(device)
elif 0: # Trained with entropy penalty to encourage attention to TWO particles, no bottleneck
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250510-123406_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth"
    EXCLUDE_TAG = False
    HAS_MLP=True
    model = TestNetwork(
        hidden_dim_attn=None, 
        hidden_dim=152, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=None, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=3,
        hidden_dim_mlp=400, 
        num_heads=4,
        embedding_size=N_CTX, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX,
        num_object_net_layers=1,
        is_layer_norm=False
    ).to(device)
elif 0: # Trained WITHOUT entropy penalty to encourage attention to a specific number of particles, no bottleneck
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250510-123629_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth"
    EXCLUDE_TAG = False
    HAS_MLP=True
    model = TestNetwork(
        hidden_dim_attn=None, 
        hidden_dim=152, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=None, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=3,
        hidden_dim_mlp=400, 
        num_heads=4,
        embedding_size=N_CTX, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX,
        num_object_net_layers=1,
        is_layer_norm=False
    ).to(device)
elif 0: # Trained WITH entropy penalty to encourage attention to a SINGLE particle, no bottleneck
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250512-093602_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_54830.pth"
    EXCLUDE_TAG = False
    HAS_MLP=True
    model = TestNetwork(
        hidden_dim_attn=None, 
        hidden_dim=152, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=None, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=3,
        hidden_dim_mlp=400, 
        num_heads=4,
        embedding_size=N_CTX, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX,
        num_object_net_layers=1,
        is_layer_norm=False
    ).to(device)
elif 0: # Trained WITH entropy penalty to encourage attention to a SINGLE particle, bottleneck of size 1
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250512-093806_TrainingOutput/models/Nplits2_ValIdx0/chkpt9_54830.pth"
    EXCLUDE_TAG = False
    HAS_MLP=True
    model = TestNetwork(
        hidden_dim_attn=None, 
        hidden_dim=152, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=1, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=6,
        hidden_dim_mlp=400, 
        num_heads=4,
        embedding_size=N_CTX, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX,
        num_object_net_layers=1,
        is_layer_norm=False
    ).to(device)
elif 1: # Trained WITH entropy penalty to encourage attention to a SINGLE particle, bottleneck of size 1
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250512-093728_TrainingOutput/models/Nplits2_ValIdx0/chkpt29_164490.pth"
    EXCLUDE_TAG = False
    HAS_MLP=True
    model = TestNetwork(
        hidden_dim_attn=None, 
        hidden_dim=152, 
        feature_set=['phi', 'eta', 'pt', 'm'] + ['tag']*(not EXCLUDE_TAG), 
        bottleneck_attention=1, 
        include_mlp=HAS_MLP, 
        num_attention_blocks=3,
        hidden_dim_mlp=400, 
        num_heads=4,
        embedding_size=N_CTX, 
        num_classes=3, 
        use_lorentz_invariant_features=True, 
        dropout_p=0.0, 
        num_particle_types=N_CTX,
        num_object_net_layers=1,
        is_layer_norm=False
    ).to(device)
# Load up the model
loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
model.load_state_dict(loaded_state_dict)
# If the model has a bottleneck, we need to hook the attention heads
if model.bottleneck_attention is not None:
    fwd_hooks = []
    hooks = []
    dummy_cache = ActivationCache()
    fwd_hooks.extend(hook_attention_heads(model, dummy_cache, detach=True, SINGLE_ATTENTION=False, bottleneck_attention_output=model.bottleneck_attention))
    for module, hook_fn in fwd_hooks:
        hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
model.eval()

# %% Evaluate the model with standard inference
val_metrics_MCWts = HEPMetrics(
    N_CTX-1, 
    max_n_objs_to_read, 
    is_categorical=IS_CATEGORICAL, 
    num_categories=3, 
    max_bkg_levels=[100, 200], 
    max_buffer_len=int(val_dataloader.get_total_samples()), 
    total_weights_per_dsid=val_dataloader.weight_sums, 
    signal_acceptance_levels=[100, 500, 1000, 5000]
)

val_dataloader._reset_indices()
val_metrics_MCWts.reset()
criterion = HEPLossWithEntropy(entropy_loss=True, entropy_weight=1e-2, target_entropy=0, is_categorical=IS_CATEGORICAL, apply_correlation_penalty=False, alpha=1.0, apply_valid_penalty=False, valid_penalty_weight=1.0)

if 1:
    model.eval()
    num_samps_to_eval = 30000
    num_batches_to_process = int(num_samps_to_eval * (1/batch_size))
    # num_batches_to_process = len(val_dataloader)
    num_batches_to_process = 1
    for batch_idx in range(num_batches_to_process):
        if ((batch_idx%10)==9):
            print(f"Processing batch {batch_idx}/{num_batches_to_process}")
        batch = next(val_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
        if 0:  # See what it's like if we randomly guess
            outputs = torch.rand(x.shape[0], x.shape[1], 3)
        # outputs, cache = mechinterputils.run_with_cache_and_bottleneck(model, x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
        outputs = outputs.squeeze()
        val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types)
    
    # Only get the loss on one final batch (limit of our current loss calculation code)
    model_activation_extractor = ModelActivationExtractor(model)
    cache = model_activation_extractor.extract_activations((x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types))
    loss, loss_dict = criterion(cache, outputs, x[...,-1], types, N_CTX-1, max_n_objs_to_read, w, False, x[...,:4])
    
    rv = val_metrics_MCWts.compute_and_log(1, 'val', 0, 3, False, None, calc_all=True)

    # Print specific reconstruction percentages for different categories
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
# Temporary cell to comparison between two models (requires loading two models) for a single batch element
# comparting the attention weight patterns for a specific batch element, layer, head
if 0:
    batch_idx=0
    for cache, title in zip([cache0, cache1, cache2, cache1bn], ["Without attention forcing", "Single attention forcing", "Double attention forcing", "Single attention forcing with bottleneck"]):
        num_layers = len([k for k in cache.store.keys() if (('attention' in k) and (not ('post' in k)))])
        num_heads = cache[f'block_{0}_attention']['attn_weights_per_head'].shape[1]
        plt.figure(figsize=(4*num_heads, 4*num_layers))
        for layer in range(num_layers):
            for head in range(num_heads):
                plt.subplot(num_layers, num_heads, layer*num_heads + head + 1)
                plt.imshow(cache[f'block_{layer}_attention']['attn_weights_per_head'][batch_idx,head][types[batch_idx]!=N_CTX-1][:, types[batch_idx]!=N_CTX-1])
                plt.title(f"layer {layer}, head {head}")
                plt.colorbar(fraction=0.046, pad=0.04)
        plt.suptitle(f"Event {batch_idx}, {title}: {print_inclusion(x[batch_idx,:,-1], types[batch_idx])}")
        plt.tight_layout()
        plt.show()
        plt.close()

# %% Check the incorrect-reconstruction rate
if 0:
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
            print(f"Processing batch {batch_idx}/{num_batches_to_process}")
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
        print(f"{DSID_MASS_MAPPING[dsid]}: lvbb mis-reco = {misreco_rate_lvbb:5.2f}, qqbb mis-reco = {misreco_rate_qqbb:5.2f}, all mis-reco = {misreco_rate_all:5.2f}")

# %%
# Evaluate the model BUT switch the last small-R jet to a neutrino or small-R jet to see how much it messes stuff up
# Predictably, it now does really badly (esp at qqbb)
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
        types_new = types.clone()
        if 0:
            types_new[torch.arange(types.shape[0]),last_sjet_index] = 2 # Replace with neutrino
        else:
            types_new[torch.arange(types.shape[0]),last_sjet_index] = 3 # Replace with large-R jet

        outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types_new).squeeze()
        val_metrics_MCWts.update(outputs, x[...,-1], MCWts, dsids, types_new)
        val_metrics_MCWts2.update(outputs, x[...,-1], MCWts, dsids, types)

    rv_messedup=val_metrics_MCWts.compute_and_log(1,'val', 0, 3, False, None)
    rv_messedup2=val_metrics_MCWts2.compute_and_log(1,'val', 0, 3, False, None)

    # Print specific reconstruction percentages for different categories
    print("New scores: with last small-R jet replaced with large-R jet")
    for k in rv_messedup.keys():
        if 'tRecoPct_all_cat' in k:
            print(f"{k:20s}: {rv_messedup[k]:.4f}")
    print("--------------------------------")
    print("New scores: with last small-R jet still counted as small-R jet")
    for k in rv_messedup2.keys():
        if 'tRecoPct_all_cat' in k:
            print(f"{k:20s}: {rv_messedup2[k]:.4f}")
    print("--------------------------------")
    print("Original scores:")
    for k in rv.keys():
        if 'tRecoPct_all_cat' in k:
            print(f"{k:20s}: {rv[k]:.4f}")



# %%
if 0:
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


# %% Do some attention analysis
if 1:
    model_activation_extractor = ModelActivationExtractor(model)
    COMBINE_LEPTONS_FOR_PLOTS=False
    cache = model_activation_extractor.extract_activations((batch['x'][...,:N_Real_Vars-int(EXCLUDE_TAG)], batch['types']))

    attention_analyzer = AttentionAnalyzer(model, cache)
    ota = attention_analyzer.analyze_type_attention(types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS)
    ota_selfex = attention_analyzer.analyze_type_attention(types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS, exclude_self=True)
    # and now plot using the visualization methods
    fig = attention_analyzer.visualize_type_attention(ota, {0: 'e', 1: 'm', 2: 'n', 3: 'l', 4: 's'}, layer_range=range(len(model.attention_blocks)), head_range=range(model.attention_blocks[0].self_attention.num_heads))
    fig.savefig('tmpAverageAttn.png')
    fig.show()
    fig = attention_analyzer.visualize_type_attention(ota_selfex, {0: 'e', 1: 'm', 2: 'n', 3: 'l', 4: 's'}, layer_range=range(len(model.attention_blocks)), head_range=range(model.attention_blocks[0].self_attention.num_heads))
    fig.savefig('tmpAverageAttnSelfEx.png')
    fig.show()
    # TODO: Do the same thing but selecting per event_category (and for correctly reconstructed/all events)
    # Will require looping over dataloader to get enough events for some of the categories

# %% 
# Exactly the same as above but now we separate out per (true) event reconstruction category
# and by successful/failed reconstruction
# Loop over the dataloader and collect activations until we get enough events for each category
# Then do the same analysis as above
if 1:
    COMBINE_LEPTONS_FOR_PLOTS=False
    for category in [5,4,3,2,1,0]:
        for success in [True, False]:
            min_events = 1000
            val_dataloader._reset_indices() # Ensure you get a fresh batch
            n_events_passing = 0
            input_events = []
            input_types = []
            targets = []
            for batch_idx in range(len(val_dataloader)-3):
                batch = next(val_dataloader)
                x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                event_categories = check_category(types, x[...,-1], padding_token, use_torch=True)
                outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
                truths = (x[...,-1]>=2)*2 + (x[...,-1]==1)
                perfect_reco = ((outputs.argmax(dim=-1) == truths) | (types==padding_token)).all(dim=-1)
                mask = (event_categories==category) & (perfect_reco == success)
                if mask.sum() == 0:
                    continue
                input_events.append(batch['x'][mask, ..., :N_Real_Vars-int(EXCLUDE_TAG)])
                input_types.append(types[mask])
                targets.append(batch['x'][mask, ..., -1])
                n_events_passing += mask.sum()
                if n_events_passing >= min_events:
                    break
                if (batch_idx % 10)==9:
                    print(f"Processing batch {batch_idx}")
            if n_events_passing < min_events:
                print(f"Only {n_events_passing} events passing for category {category} and success {success}")
                continue
            input_events = torch.cat(input_events, dim=0)
            input_types = torch.cat(input_types, dim=0)
            targets = torch.cat(targets, dim=0)
            print(f"Found {n_events_passing} events passing for category {category} and success {success}")
            # Now do the same analysis as above
            # First, get the activations
            model_activation_extractor = ModelActivationExtractor(model)
            cache = model_activation_extractor.extract_activations((input_events, input_types))
            attention_analyzer = AttentionAnalyzer(model, cache)
            ota = attention_analyzer.analyze_type_attention(input_types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS)
            ota_selfex = attention_analyzer.analyze_type_attention(input_types, padding_token, combine_elec_and_muon=COMBINE_LEPTONS_FOR_PLOTS, exclude_self=True)
            # and now plot using the visualization methods
            fig = attention_analyzer.visualize_type_attention(ota, {0: 'e', 1: 'm', 2: 'n', 3: 'l', 4: 's', 5:'p'}, layer_range=range(len(model.attention_blocks)), head_range=range(model.attention_blocks[0].self_attention.num_heads))
            fig.savefig(f'tmpAverageAttn2_cat{category}_success{success}.png')
            plt.title(f"Event category {category}, success {success}")
            fig.show()
            fig = attention_analyzer.visualize_type_attention(ota_selfex, {0: 'e', 1: 'm', 2: 'n', 3: 'l', 4: 's', 5:'p'}, layer_range=range(len(model.attention_blocks)), head_range=range(model.attention_blocks[0].self_attention.num_heads))
            fig.savefig(f'tmpAverageAttn2SelfEx_cat{category}_success{success}.png')
            plt.title(f"Event category {category}, success {success}")
            fig.show()


# %% Temporary cell to choose an interesting event to analyze. Look at the true and predicted 
# inclusion of the event to choose one, then plot the attention weights for the event
if 1: 
    if 1: # To choose an event to analyze, look at the true and predicted inclusion of the event
        val_dataloader._reset_indices()
        batch = next(val_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
        for batch_idx in range(10, 20):
            print(f"Event {batch_idx}: ")
            print(f"\t True inclusion: {print_inclusion(x[batch_idx,:,-1], types[batch_idx])}")
            print(f"\t Pred inclusion: {print_inclusion(outputs.argmax(dim=-1)[batch_idx], types[batch_idx])}")
    if 0: # Plot the attention weights for the event
        batch_idx=16
        num_layers = len([k for k in cache.store.keys() if (('attention' in k) and (not ('post' in k)))])
        num_heads = cache[f'block_{0}_attention']['attn_weights_per_head'].shape[1]
        plt.figure(figsize=(4*num_heads, 4*num_layers))
        _, event_objs_list = print_inclusion(x[batch_idx,:,-1], types[batch_idx], return_array=True)
        for layer in range(num_layers):
            for head in range(num_heads):
                plt.subplot(num_layers, num_heads, layer*num_heads + head + 1)
                plt.imshow(cache[f'block_{layer}_attention']['attn_weights_per_head'][batch_idx,head][types[batch_idx]!=N_CTX-1][:, types[batch_idx]!=N_CTX-1])
                plt.title(f"layer {layer}, head {head}")
                # plt.xticks(range(len(event_objs_list)), event_objs_list, rotation=90)
                plt.xticks(range(len(event_objs_list)), event_objs_list)
                plt.yticks(range(len(event_objs_list)), event_objs_list)
                plt.xlabel('Key Object')
                plt.ylabel('Query Object')
                plt.colorbar(fraction=0.046, pad=0.04)
        plt.suptitle(f"Event {batch_idx}, {title}: {print_inclusion(x[batch_idx,:,-1], types[batch_idx])}")
        plt.tight_layout()
        plt.show()
        plt.close()


# %% Example: Direct Logit Attribution for one event
import importlib
import sys
def m_reload():
    for k,v in sys.modules.items():
        if k.startswith('interp'):
            importlib.reload(v)
# Initialize Patcher and Attributor
# Ensure model_activation_extractor is initialized before this
activation_patcher = ActivationPatcher(model, model_activation_extractor)
direct_logit_attributor = DirectLogitAttributor(model, model_activation_extractor, activation_patcher)

val_dataloader._reset_indices() # Ensure you get a fresh batch
batch = next(val_dataloader)
outputs = model(batch['x'][...,:N_Real_Vars-int(EXCLUDE_TAG)], batch['types']).squeeze()
# Select a single event from the batch for focused analysis, or adapt for batch processing
event_features = batch['x'][0:1, ..., :N_Real_Vars-int(EXCLUDE_TAG)] # Batch size 1
event_types = batch['types'][0:1, ...]
event_truth_labels = (batch['x'][0:1, ..., -1] > 1) * 2 + (batch['x'][0:1, ..., -1] == 1)
event_predicted_labels = outputs.argmax(dim=-1)
inputs_for_analysis = (event_features, event_types)

target_class_idx = 2  # Example: class for W-boson products
target_object_idx = 0 # Example: first object in the event (if reconstruction model)
batch_idx_to_analyze = 0 # Since we took a slice of size 1

# Define components you want to analyze. These strings need to match what _get_ablation_patch_ops expects.
components = [
    "object_net_output",
    "all_mha_outputs",        # Ablates output of each MHA module
    "all_attention_heads",    # Ablates each head's contribution individually
    "all_mlp_outputs"         # Ablates output of each post-attention MLP
]
if not model.include_mlp:
    components.remove("all_mlp_outputs")


print(f"Analyzing DLA for event 0, target class {target_class_idx}, target object {target_object_idx}")
dla_results = direct_logit_attributor.analyze_logit_attribution(
    inputs_tuple=inputs_for_analysis,
    target_class_idx=target_class_idx,
    components_to_ablate=components,
    batch_idx=batch_idx_to_analyze, # Index within the provided inputs_tuple
    target_object_idx=target_object_idx if model_activation_extractor.model_type == 'reconstruction' else None
)

print("\nAblation analysis Logit Attribution Results:")
for comp, attr in dla_results.items():
    print(f"  {comp}: {attr:.4f}")

# %% Example: Activation Patching (Causal Tracing)
# Suppose you want to see how attention block 0 from a 'clean_event'
# affects a 'corrupted_event' if patched in.

# Create dummy clean and corrupted inputs (replace with actual data)
# For simplicity, let's use the same event as source and original, but patch a zero tensor
# to demonstrate patching an arbitrary value.
# clean_inputs = inputs_for_analysis
# corrupted_inputs = inputs_for_analysis 

# Patch operation: Replace output of attention_blocks.0.self_attention with zeros
# (This is essentially what DLA does for this component)
if 0: # basic example
    block_num = 3
    head_num = 0
    patch_op_zero_mha0 = {
        'target_module_path': f'attention_blocks.{block_num}.self_attention',
        # 'patch_value_fn': lambda src_cache, orig_out_tuple: (torch.zeros_like(orig_out_tuple[0]), orig_out_tuple[1]),
        'patch_value_fn': lambda src_cache, orig_out_tuple: torch.zeros_like(orig_out_tuple[0]),
        'target_output_component_idx': 0,
        'head_idx': head_num
    }

    print(f"\nRunning with MHA block {block_num} output zeroed (example of patching):")
    patched_output_mha0_zeroed = activation_patcher.patch_and_run(
        original_inputs_tuple=inputs_for_analysis,
        patch_operations=[patch_op_zero_mha0],
        source_inputs_tuple=None # No source_cache needed for this specific patch_value_fn
    )
    # print(patched_output_mha0_zeroed.to(int))
    # print(outputs[0:1].to(int))
    print("Types:")
    print(batch['types'][0:1].to(int))
    print(f"Change in logit for target class {target_class_idx}:")
    print((patched_output_mha0_zeroed - outputs[0:1])[batch['types'][0:1]!=padding_token])

    # You would then compare patched_output_mha0_zeroed with the clean output.



# For actual causal tracing, source_inputs_tuple would be different from original_inputs_tuple,
# and patch_value_fn would typically extract an activation from source_cache.
# Example:

val_dataloader._reset_indices() # Ensure you get a fresh batch
batch = next(val_dataloader)
outputs = model(batch['x'][...,:N_Real_Vars-int(EXCLUDE_TAG)], batch['types']).squeeze()
# Select a single event from the batch for focused analysis, or adapt for batch processing
event_features = batch['x'][0:1, ..., :N_Real_Vars-int(EXCLUDE_TAG)] # Batch size 1
event_types = batch['types'][0:1, ...]
event_truth_labels = (batch['x'][0:1, ..., -1] > 1) * 2 + (batch['x'][0:1, ..., -1] == 1)
event_predicted_labels = outputs.argmax(dim=-1)
inputs_for_analysis = (event_features, event_types)

clean_inputs = inputs_for_analysis
corrupted_types = event_types.clone()
corrupted_types[0, batch['types'][0]==4] = 3 # Tell the event that the small-R jets are large-R jets
corrupted_inputs = (event_features, corrupted_types)
# patch_op_trace_head0 = {
#    'target_module_path': 'attention_blocks.0.self_attention',
#    'patch_value_fn': lambda src_cache, orig_out_tuple: 
#        src_cache['block_0_attention']['attn_output_per_head'][:, 0, :, :].sum(dim=0, keepdim=True) # Simplified: sum head 0 output
#         #   .expand_as(orig_out_tuple[0]) # This reconstruction is too simple, proper one needed
#    'target_output_component_idx': 0
# }
block_num = 0
head_num = 0
patch_op_trace_head0 = {
   'target_module_path': f'attention_blocks.{block_num}.self_attention',
   'patch_value_fn': lambda src_cache, orig_out_tuple: 
       # Sum the original heads EXCEPT head 0 with the patched head 0 output
       orig_out_tuple[0][:, [i for i in range(orig_out_tuple[0].shape[1]) if i != head_num], :, :].sum(dim=1) + \
       src_cache['block_0_attention']['attn_output_per_head'][:, head_num, :, :] + \
       model.attention_blocks[block_num].out_proj.bias,
   'target_output_component_idx': 0 # For attention, the output is a tuple of (attn_output, attn_weights)
   # and we want to change the attn_output but don't care about the attn_weights
}
patched_output_traced = activation_patcher.patch_and_run(
   original_inputs_tuple=corrupted_inputs, # e.g., event with small-R jets labelled as large-R jets
   patch_operations=[patch_op_trace_head0],
   source_inputs_tuple=clean_inputs # e.g., original event
)

print("Types:")
print(batch['types'][0:1].to(int))
print(f"Change in logit for target class {target_class_idx}:")
print((patched_output_traced - outputs[0:1])[batch['types'][0:1]!=padding_token])



# %%
direct_contrib_analyzer = DirectLogitContributionAnalyzer(model, model_activation_extractor)

# %% Example: Direct Logit Contribution analysis
# (Assuming inputs_for_analysis, target_class_idx, batch_idx_to_analyze, target_object_idx are defined)

components_for_direct_contrib = [
    {'name': 'object_net_output'},
    {'name': 'mha_output', 'block': 0},
    {'name': 'head_output', 'block': 0, 'head': 0},
    {'name': 'head_output', 'block': 0, 'head': 1},
    # Add more heads or blocks
]
if model.include_mlp:
    components_for_direct_contrib.append({'name': 'mlp_output', 'block': 0})


print(f"\nAnalyzing Direct Logit Contributions for event {batch_idx_to_analyze}, target class {target_class_idx}" +
      (f", target object {target_object_idx}" if model_activation_extractor.model_type == 'reconstruction' or target_object_idx is not None else ", averaged over objects"))

direct_contributions = direct_contrib_analyzer.analyze_contribution(
    inputs_tuple=inputs_for_analysis,
    target_class_idx=target_class_idx,
    components_to_analyze=components_for_direct_contrib,
    batch_idx=batch_idx_to_analyze,
    target_object_idx=target_object_idx, # Or None for classification object averaging
    include_bias_in_contribution=False # Often better to analyze bias separately
)

print("\nDirect Logit Contribution Results:")
for comp_name, contrib_val in direct_contributions.items():
    print(f"  {comp_name}: {contrib_val:.4f}")





# %%
# 1. Extract activations and residuals
extractor = ModelActivationExtractor(model, model_type='reconstruction')
cache = extractor.extract_activations((batch['x'][..., :N_Real_Vars-int(EXCLUDE_TAG)], batch['types']))
residuals = get_residual_stream(cache, verbose=True)  # Dict: {0: ..., 1: ..., ...}

# 2. Select the layer/block you want to analyze
for block_idx in residuals.keys():
    # block_idx = 'object_net'  # e.g., after first attention block
    resid = residuals[block_idx] # shape: [batch, n_obj, d_model]

    # 3. Select objects of interest (e.g., large-R jets with truth label 2)
    is_ljet = (batch['types'] == 3)
    is_sjet = (batch['types'] == 4)
    is_lep = (batch['types'] == 0) | (batch['types'] == 1)
    is_truth_W = (batch['x'][..., -1] >= 2)
    # mask = is_ljet # & is_truth_W  # [batch, n_obj]
    # mask = is_lep
    mask = is_sjet
    # 4. Flatten and probe
    X = flatten_selected_objects(resid, mask)
    y = batch['x'][..., -1][mask].cpu().numpy()
    
    # Split into train/val sets
    n_samples = len(X)
    # train_size = int(0.8 * n_samples)
    train_size = min(int(0.8 * n_samples), 2000)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train and evaluate
    clf = fit_linear_probe(X_train, y_train, task='classification', max_iter=5000)
    train_acc = probe_accuracy(clf, X_train, y_train, task='classification')
    val_acc = probe_accuracy(clf, X_val, y_val, task='classification')
    print(f"Probe accuracy for W-boson class after block {block_idx}: train={train_acc:.3f}, val={val_acc:.3f}")

    # plot_pca(X, y, title=f'PCA of residual stream (block {block_idx}) for large-R jets')
    # plt.close('all')





# %%
# Now simple task to check we're doing the residual stream interpretation correctly
# We will try to predict what type of object we are looking at from the residual stream
# 1. Extract activations and residuals
if 1:
    extractor = ModelActivationExtractor(model, model_type='reconstruction')
    cache = extractor.extract_activations((batch['x'][..., :N_Real_Vars-int(EXCLUDE_TAG)], batch['types']))
    residuals = get_residual_stream(cache, verbose=True)  # Dict: {0: ..., 1: ..., ...}

    # 2. Loop over all points in the residual stream and try to predict the object type
    for block_idx in residuals.keys():
        resid = residuals[block_idx] # shape: [batch, n_obj, d_model]

        # 3. Select objects of interest (e.g., large-R jets with truth label 2)
        is_in_event = (batch['x'][..., -1] >= 1)
        mask = ~is_in_event
        # 4. Flatten and probe
        X = flatten_selected_objects(resid, mask)
        y = batch['types'][mask].cpu().numpy()
        
        # Split into train/val sets
        n_samples = len(X)
        # train_size = int(0.8 * n_samples)
        train_size = min(int(0.8 * n_samples), 2000)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train and evaluate
        clf = fit_linear_probe(X_train, y_train, task='classification', max_iter=5000)
        train_acc = probe_accuracy(clf, X_train, y_train, task='classification')
        val_acc = probe_accuracy(clf, X_val, y_val, task='classification')
        print(f"Probe accuracy for object type after block {block_idx}: train={train_acc:.3f}, val={val_acc:.3f}")

        # plot_pca(X, y, title=f'PCA of residual stream (block {block_idx}) for object type')





# %%
# Now we're going to look at events with exactly 2 large-R jets see if the residual stream
# contains information about the angle between them using linear probes. We'll use the same 
# technique as before and look at the residual stream for the first large-R jet.
if 1:
    # First, we'll loop over batches and get at least 5000 events with exactly 2 large-R jets.

    val_dataloader._reset_indices() # Ensure you get a fresh batch
    n_events_with_2_ljets = 0
    input_events = []
    input_types = []
    targets = []
    for batch_idx in range(100): # Try up to 100 batches
        batch = next(val_dataloader)
        types = batch['types']
        is_ljet = (types == 3)
        num_ljets = is_ljet.sum(dim=1)
        event_categories = check_category(types, batch['x'][..., -1], padding_token, use_torch=True)
        mask = (num_ljets == 2) & (event_categories==5)
        if mask.sum() == 0:
            continue
        input_events.append(batch['x'][mask, ..., :N_Real_Vars-int(EXCLUDE_TAG)])
        input_types.append(types[mask])
        targets.append(batch['x'][mask, ..., -1])
        n_events_with_2_ljets += mask.sum()
        if n_events_with_2_ljets >= 5000:
            break
    print(f"Found {n_events_with_2_ljets} events with exactly 2 large-R jets")
    input_events = torch.cat(input_events)
    input_types = torch.cat(input_types)
    targets = torch.cat(targets)

    # Now we'll extract the residual stream for the first large-R jet in each event.
    extractor = ModelActivationExtractor(model, model_type='reconstruction')
    cache = extractor.extract_activations((input_events, input_types))
    residuals = get_residual_stream(cache, verbose=True)  # Dict: {0: ..., 1: ..., ...}

    # Now calculate the angle between the two large-R jets in each event.
    delta_rs = []
    delta_phis = []
    delta_etas = []
    for i in range(len(input_events)):
        event = input_events[i]
        types = input_types[i]
        is_ljet = (types == 3)
        ljet_indices = is_ljet.nonzero().squeeze()
        if len(ljet_indices) != 2:
            continue
        ljet1 = event[ljet_indices[0], :4]
        ljet2 = event[ljet_indices[1], :4]
        # Now calculate the delta R between the two jets
        delta_r, delta_phi, delta_eta = deltaR(ljet1, ljet2)
        delta_phis.append(delta_phi)
        delta_etas.append(delta_eta)
        delta_rs.append(delta_r)

    # Now we'll fit a linear probe to predict the angle from the residual stream at each
    # layer of the model.
    for block_idx in residuals.keys():
        # Get a mask of ONLY the first large-R jet in each event.
        is_ljet = (input_types == 3) # This is for getting ANY large-R jet, now need to get the first one
        # Can use the fact that ljets will come successively since we didn't shuffle the objects within the event
        # so can just check if the object type matches the one before it (need to also accocunt for the off-by-one
        # size error if we do this)
        is_same = (input_types[:, 1:] != input_types[:, :-1])
        is_same = torch.cat([is_same, torch.zeros((len(is_same), 1)).to(bool)], dim=1) # Now account for the off-by-one size error
        mask = is_ljet & is_same

        X = flatten_selected_objects(residuals[block_idx], mask)
        for angle_name, angles in zip(['phi', 'eta', 'R'], [delta_phis, delta_etas, delta_rs]):
            y = np.array(angles)

            # Split into train/val sets
            n_samples = len(X)
            # train_size = int(0.8 * n_samples)
            train_size = min(int(0.8 * n_samples), 2000)
            indices = np.random.permutation(n_samples)
            train_idx, val_idx = indices[:train_size], indices[train_size:]

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train and evaluate
            clf = fit_linear_probe(X_train, y_train, task='regression', max_iter=5000)
            train_acc = probe_accuracy(clf, X_train, y_train, task='regression')
            val_acc = probe_accuracy(clf, X_val, y_val, task='regression')
            print(f"Probe accuracy for ljet-pair delta-{angle_name:6s} after block {block_idx:30s}: train={train_acc:.3f}, val={val_acc:.3f}")
        print('----------------------------')




# %%
if 1:
    # Now we're going to look at events which are reconstructed as small-R jet W pair, large-R SM Higgs
    # and see if we can regress things about the W pair from looking at one or the other 'true-W' particles

    if 1:
        # First, we'll loop over batches and get at least eg. 5000 events satisfying our criteria
        min_events = 2000
        val_dataloader._reset_indices() # Ensure you get a fresh batch
        n_events_passing = 0
        input_events = []
        input_types = []
        targets = []
        for batch_idx in range(len(val_dataloader)-3): # Try up to 100 batches
            batch = next(val_dataloader)
            types = batch['types']
            event_categories = check_category(types, batch['x'][...,-1], padding_token, use_torch=True)
            outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
            truths = (batch['x'][...,-1]>=2)*2 + (batch['x'][...,-1]==1)
            perfect_reco = ((outputs.argmax(dim=-1) == truths) | (types==padding_token)).all(dim=-1)
            mask = (event_categories==3) & perfect_reco
            if mask.sum() == 0:
                continue
            input_events.append(batch['x'][mask, ..., :N_Real_Vars-int(EXCLUDE_TAG)])
            input_types.append(types[mask])
            targets.append(batch['x'][mask, ..., -1])
            n_events_passing += mask.sum()
            if n_events_passing >= min_events:
                break
            if (batch_idx % 10)==9:
                print(f"Processing batch {batch_idx}")
        print(f"Found {n_events_passing} events with exactly 2 large-R jets")
        input_events = torch.cat(input_events)
        input_types = torch.cat(input_types)
        targets = torch.cat(targets)

        # Now we'll extract the residual stream for the first large-R jet in each event.
        extractor = ModelActivationExtractor(model, model_type='reconstruction')
        cache = extractor.extract_activations((input_events, input_types))
        residuals = get_residual_stream(cache, verbose=True)  # Dict: {0: ..., 1: ..., ...}

    # Now calculate the angle between the two large-R jets in each event.
    delta_rs = []
    delta_phis = []
    delta_etas = []
    combined_masses = []
    for i in range(len(input_events)):
        event = input_events[i]
        types = input_types[i]
        target = targets[i]
        is_in_Whad = (target == 2)
        is_lepton = (types==0)|(types==1)
        Whad_indices = is_in_Whad.nonzero().squeeze()
        lepton_indices = is_lepton.nonzero().squeeze()
        if len(Whad_indices) != 2:
            continue
        sjet1 = event[Whad_indices[0], :4]
        if 0:
            sjet2 = event[Whad_indices[1], :4]
        else:
            # print("WARNING: Doing it with lepton instead!")
            sjet2 = event[lepton_indices.item(), :4]
        # Now calculate the delta R between the two jets
        delta_r, delta_phi, delta_eta = deltaR(sjet1, sjet2)
        combined_mass = Get_PtEtaPhiM_fromXYZT(*[sjet1[i]+sjet2[i] for i in range(4)], use_torch=True)[3]
        delta_phis.append(delta_phi)
        delta_etas.append(delta_eta)
        delta_rs.append(delta_r)
        combined_masses.append(combined_mass)

    # Now we'll fit a linear probe to predict the angle from the residual stream at each
    # layer of the model.
    for block_idx in residuals.keys():
        select_only_one_sjet = False
        if select_only_one_sjet:
            raise NotImplementedError
        else:
            # Get a mask of BOTH jets in the Whad. Need to think of something clever if we just want to look at one of them
            is_in_Whad = (targets == 2)
            mask = is_in_Whad

        X = flatten_selected_objects(residuals[block_idx], mask)
        for regression_target_name, regression_target in zip(['phi', 'eta', 'R', 'combined_mass'], [delta_phis, delta_etas, delta_rs, combined_masses]):
            y = np.array(regression_target)
            if not select_only_one_sjet:
                y = einops.repeat(np.array(regression_target), 'batch -> batch 2').flatten()

            # Split into train/val sets
            n_samples = len(X)
            # train_size = int(0.8 * n_samples)
            train_size = min(int(0.8 * n_samples), 2000)
            indices = np.random.permutation(n_samples)
            train_idx, val_idx = indices[:train_size], indices[train_size:]

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train and evaluate
            clf = fit_linear_probe(X_train, y_train, task='regression', max_iter=5000)
            train_acc = probe_accuracy(clf, X_train, y_train, task='regression')
            val_acc = probe_accuracy(clf, X_val, y_val, task='regression')
            print(f"Probe accuracy for ljet-pair delta-{regression_target_name:20s} after block {block_idx:30s}: train={train_acc:.3f}, val={val_acc:.3f}")
        print('----------------------------')




# %%
if 1:
    # Now we're going to look at events which are reconstructed as large-R jet W, large-R SM Higgs
    # and see if we can regress things about the relationship between two objects (can choose
    # between TrueHiggs, TrueW, RecoWlep)
    # We'll try and make the code general enough to be used for different objects, 
    # and we can just select for the objects and sum up the 4 vectors so we can even have composite
    # objects like 'neutrino + lepton' (which would be RecoWlep) or 'small-R jet pair with highest Pt' 
    # or something. However, object1 MUST be a single object, since we need to look at the residual stream
    # for that object.
    if 0: # Can predict the combined mass pretty well, but not the delta-R (0.45 ish top score)
        object1 = 'TrueHiggs'
        object2 = 'TrueW'
    elif 1:
        object1 = 'TrueW'
        object2 = 'TrueHiggs'
    elif 0: # Can predict the delta-R pretty well! (0.8 ish top score)
        object1 = 'TrueHiggs'
        object2 = 'RecoWlep'
    elif 0:
        object1 = 'TrueW'
        object2 = 'RecoWlep'
    elif 1:
        object1 = 'TrueHiggs'
        object2 = 'HighestPtSmallRJet'
    else:
        assert(False)
    if 0:
        # First, we'll loop over batches and get at least eg. 5000 events satisfying our criteria
        min_events = 10000
        val_dataloader._reset_indices() # Ensure you get a fresh batch
        n_events_passing = 0
        input_events = []
        input_types = []
        targets = []
        for batch_idx in range(len(val_dataloader)-3): # Try up to 100 batches
            batch = next(val_dataloader)
            types = batch['types']
            event_categories = check_category(types, batch['x'][...,-1], padding_token, use_torch=True)
            outputs = model(x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types)
            truths = (batch['x'][...,-1]>=2)*2 + (batch['x'][...,-1]==1)
            perfect_reco = ((outputs.argmax(dim=-1) == truths) | (types==padding_token)).all(dim=-1)
            mask = (event_categories==5) & perfect_reco
            if mask.sum() == 0:
                continue
            input_events.append(batch['x'][mask, ..., :N_Real_Vars-int(EXCLUDE_TAG)])
            input_types.append(types[mask])
            targets.append(batch['x'][mask, ..., -1])
            n_events_passing += mask.sum()
            if n_events_passing >= min_events:
                break
            if (batch_idx % 10)==9:
                print(f"Processing batch {batch_idx}")
        print(f"Found {n_events_passing} events with large-R jet W, large-R SM Higgs")
        input_events = torch.cat(input_events)
        input_types = torch.cat(input_types)
        targets = torch.cat(targets)

        # Now we'll extract the residual stream for the first large-R jet in each event.
        extractor = ModelActivationExtractor(model, model_type='reconstruction')
        cache = extractor.extract_activations((input_events, input_types))
        residuals = get_residual_stream(cache, verbose=True)  # Dict: {0: ..., 1: ..., ...}

    # Now calculate the angle between the two large-R jets in each event.
    delta_rs = []
    delta_phis = []
    delta_etas = []
    combined_masses = []
    combined_masses_sq = []
    pt_ratios = []
    for i in range(len(input_events)):
        event = input_events[i]
        types = input_types[i]
        target = targets[i]
        is_in_Whad = (target == 2)
        is_in_Higgs = (target == 1)
        is_lepton = (types==0)|(types==1)
        is_neutrino = (types==2)
        is_small_r_jet = (types==4)
        Whad_indices = is_in_Whad.nonzero().squeeze()
        Higgs_indices = is_in_Higgs.nonzero().squeeze()
        lepton_indices = is_lepton.nonzero().squeeze()
        neutrino_indices = is_neutrino.nonzero().squeeze()
        small_r_jet_indices = is_small_r_jet.nonzero().squeeze()
        if object1=='TrueHiggs':
            obj1 = event[Higgs_indices.item(), :4]
        elif object1=='TrueW':
            obj1 = event[Whad_indices.item(), :4]
        elif object1 == 'RecoWlep':
            obj1 = event[lepton_indices.item(), :4] + event[neutrino_indices.item(), :4]
        else:
            raise ValueError(f"Invalid object1: {object1}")
        if object2=='TrueHiggs':
            obj2 = event[Higgs_indices.item(), :4]
        elif object2=='TrueW':
            obj2 = event[Whad_indices.item(), :4]
        elif object2 == 'RecoWlep':
            obj2 = event[lepton_indices.item(), :4] + event[neutrino_indices.item(), :4]
        elif object2 == 'HighestPtSmallRJet':
            # Need to get the index of the highest pt small-R jet
            pt_small_r_jets = Get_PtEtaPhiM_fromXYZT(*event[small_r_jet_indices, :4], use_torch=True)[0]
            obj2 = event[small_r_jet_indices[pt_small_r_jets.argmax()], :4]
        else:
            raise ValueError(f"Invalid object2: {object2}")
        # Now calculate the delta R between the two objects
        delta_r, delta_phi, delta_eta = deltaR(obj1, obj2)
        ptetaphim1 = Get_PtEtaPhiM_fromXYZT(*obj1, use_torch=True)
        ptetaphim2 = Get_PtEtaPhiM_fromXYZT(*obj2, use_torch=True)
        combined_mass = Get_PtEtaPhiM_fromXYZT(*[obj1[i]+obj2[i] for i in range(4)], use_torch=True)[3]
        delta_phis.append(delta_phi.abs())
        delta_etas.append(delta_eta.abs())
        delta_rs.append(delta_r)
        combined_masses.append(combined_mass)
        combined_masses_sq.append(combined_mass**2)
        pt_ratios.append(ptetaphim1[0]/ptetaphim2[0])

    # Now we'll fit a linear probe to predict the angle from the residual stream at each
    # layer of the model.
    for block_idx in residuals.keys():
        is_in_Higgs = (targets == 1)
        is_in_Whad = (targets == 2)
        is_lepton = (input_types==0)|(input_types==1)
        is_neutrino = (input_types==2)
        if 0: # Get both objects
            if object1 == 'RecoWlep':
                is_obj1 = is_lepton | is_neutrino
            elif object1 == 'TrueW':
                is_obj1 = is_in_Whad
            elif object1 == 'TrueHiggs':
                is_obj1 = is_in_Higgs
            else:
                raise ValueError(f"Invalid object1: {object1}")
            if object2 == 'RecoWlep':
                is_obj2 = is_lepton | is_neutrino
            elif object2 == 'TrueW':
                is_obj2 = is_in_Whad
            elif object2 == 'TrueHiggs':
                is_obj2 = is_in_Higgs
            else:
                raise ValueError(f"Invalid object2: {object2}")
            mask = is_obj1 | is_obj2
        else: # Just look at object1
            if object1 == 'RecoWlep':
                mask = is_lepton | is_neutrino
            elif object1 == 'TrueW':
                mask = is_in_Whad
            elif object1 == 'TrueHiggs':
                mask = is_in_Higgs
            else:
                raise ValueError(f"Invalid object1: {object1}")

        X = flatten_selected_objects(residuals[block_idx], mask)
        for regression_target_name, regression_target in zip(['phi', 'eta', 'R', 'combined_mass', 'combined_mass_sq', 'pt_ratio'], [delta_phis, delta_etas, delta_rs, combined_masses, combined_masses_sq, pt_ratios]):
            y = np.array(regression_target)

            # Split into train/val sets
            n_samples = len(X)
            # train_size = int(0.8 * n_samples)
            train_size = min(int(0.8 * n_samples), 2000)
            indices = np.random.permutation(n_samples)
            train_idx, val_idx = indices[:train_size], indices[train_size:]

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train and evaluate
            clf = fit_linear_probe(X_train, y_train, task='regression', max_iter=100000)
            train_acc = probe_accuracy(clf, X_train, y_train, task='regression')
            val_acc = probe_accuracy(clf, X_val, y_val, task='regression')
            print(f"Probe accuracy for ljet-pair delta-{regression_target_name:20s} after block {block_idx:30s}: train={train_acc:.3f}, val={val_acc:.3f}")
        print('----------------------------')

# %%
# Check the number of samples per category in a batch

val_dataloader._reset_indices() # Ensure you get a fresh batch
batch = next(val_dataloader)
event_categories = check_category(batch['types'], batch['x'][...,-1], padding_token, use_torch=True)
torch.bincount(event_categories)


# %%
if 1:
    # Now we're going to do some ablation analysis and see which event types lose performance when we ablate certain heads
    # Define a helper 'evaluate' function

    def model_eval(model, dataloader, interventions, num_samples=5000, verbose=False, report_all=True):
        metric_tracker = HEPMetrics(N_CTX-1, max_n_objs_to_read, is_categorical=IS_CATEGORICAL, num_categories=3, max_bkg_levels=[200], max_buffer_len=int(dataloader.get_total_samples()), total_weights_per_dsid=dataloader.weight_sums, signal_acceptance_levels=[100, 1000])
        dataloader._reset_indices()
        metric_tracker.reset()
        model.eval()
        model_activation_extractor = ModelActivationExtractor(model)
        num_batches_to_process = int(num_samples * (1/batch_size))
        min_per_category = 1000
        bincounts = torch.zeros((6))
        batch_idx = 0
        while (batch_idx < num_batches_to_process) or (bincounts.min() < min_per_category):
            if ((batch_idx%10)==9) and verbose:
                print(F"Processing batch {batch_idx}/{num_batches_to_process}")
            batch = next(dataloader)
            x, _, _, types, dsids, _, _, MCWts, _ = batch.values()
            event_categories = check_category(types, x[...,-1], padding_token, use_torch=True)
            bincounts += torch.bincount(event_categories)

            cache = model_activation_extractor.extract_activations((x[...,:N_Real_Vars-int(EXCLUDE_TAG)], types), interventions=interventions)
            outputs = cache['classifier']['output']
            metric_tracker.update(outputs, x[...,-1], MCWts, dsids, types)
            event_categories = check_category(batch['types'], batch['x'][...,-1], padding_token, use_torch=True)
            batch_idx+=1
            print(bincounts)
        if report_all:
            return metric_tracker.compute_and_log(1,'val', 0, 3, False, None, False, calc_all=True)
        else:
            results = metric_tracker.compute_and_log(1,'val', 0, 3, False, None, False, calc_all=False)
            summary = {k:results[k] for k in results.keys() if 'tRecoPct_all_cat' in k}
            return summary

    # Quick test
    if 0:
        layer = 5
        head = 0
        interventions = [
            (f'attention_blocks.{layer}.self_attention', lambda x: ablate_attention_head(x, head_idx=head, num_heads=model.attention_blocks[layer].self_attention.num_heads))
        ]
        results = model_eval(model, val_dataloader, interventions, report_all=False)
        print(results)
        # and without the intervention
        results_no_intervention = model_eval(model, val_dataloader, [], report_all=False)
        print(results_no_intervention)

    if 1: # Evaluate the model with different attention heads ablated
        results_single_ablation = {}
        results_single_ablation[tuple()] = model_eval(model, val_dataloader, [], report_all=True)
        print(f"None: lvbball={results_single_ablation[tuple()]['val/PerfectRecoPct_all_lvbb']:.4f} qqbball={results_single_ablation[tuple()]['val/PerfectRecoPct_all_qqbb']:.4f}")
        for layer in range(len(model.attention_blocks)):
            for head in range(model.attention_blocks[0].self_attention.num_heads):
                interventions = [
                    (f'attention_blocks.{layer}.self_attention', lambda x: ablate_attention_head(x, head_idx=head, num_heads=model.attention_blocks[0].self_attention.num_heads))
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
# Symbolic regression code
#   Here, we'll collect the outputs from the bottleneck layers (assuming that model.bottleneck_attention==1, 
#   which we should check with an assert) and try to regress what variables are being passed to/from objects,
#   as closed form expressions of sensible potential input variables
#   In order to do this, we'll select only events which pass certain criteria (which should be configurable)
#   and look at the bottleneck layer output for only certain types of object. We'd also like to have the 
#   ability to filter based on the event weight of that particular attention head, so that, for example, we 
#   can choose to only include objects which are paying more than 90% of their attention to eg. the lepton, 
#   or the true Higgs, etc. The symbolic regression inputs will be variables about the object whose 
#   bottleneck stream we are looking at (the 'query') and the object that it is paying most attention to (
#   the 'key')
#   We'll make use of existing functions, and the symbolic regression functions will be written into a new 
#   file in 'interp/symbolicRegression.py'. 
from interp.activations import extract_filtered_activations
from interp.symbolic_regression import symbolic_regression_on_bottleneck
# Define your event/object selection functions
def event_filter_fn(batch):
    # # Example: select all events
    # return np.ones(batch['x'].shape[0], dtype=bool)
    # Select all events in category 5
    return (4 == check_category(batch['types'], batch['x'][...,-1], padding_token, use_torch=True))


def object_filter_fn_query(batch, types, event_idx):
    # # Example: select all objects except padding
    # return (types != (batch['types'].max().item()))
    # Select true Higgs large-R jet
    return (batch['x'][event_idx,:, -1] == 1) & (types==3)

def object_filter_fn_key(batch, types, event_idx):
    # # Example: select all objects except padding
    # return (types != (batch['types'].max().item()))
    # Select leptons
    return (types==0)|(types==1)

# Run extraction
val_dataloader._reset_indices()
result = extract_filtered_activations(
    val_dataloader,
    model,
    n_features=N_Real_Vars-int(EXCLUDE_TAG),
    event_filter_fn=event_filter_fn,
    object_filter_fn_query=object_filter_fn_query,
    object_filter_fn_key=object_filter_fn_key,
    # activations_to_return=['block_0_attention.bottleneck_activation.1', 'block_2_attention.bottleneck_activation.2'],
    activations_to_return=['block_0_attention.bottleneck_activation.1'],
    max_samples=10000,
)
if 1:
    # Run symbolic regression
    X_query = result['query_features'].numpy()
    X_key = result.get('key_features', None)
    Y = result['activations']['block_0_attention.bottleneck_activation.1'].numpy()
    # Y2 = result['activations']['block_2_attention.bottleneck_activation.2'].numpy()
    symbolic_config = {
        'package': 'pysr',
        'niterations': 40,
        'maxsize': 30,
        'populations': 100,
        'population_size': 30,
        'binary_operators': ['+', '-', '*', '/'],
        'unary_operators': ['square', 'sqrt'],
        'variable_names': ['H_pt', 'H_eta', 'H_phi', 'H_mass', 'H_tag', 'lep_pt', 'lep_eta', 'lep_phi', 'lep_mass', 'lep_tag'],
    }
    regression_results = symbolic_regression_on_bottleneck(X_query, X_key, np.expand_dims(Y, axis=-1), regression_config=symbolic_config)
    # regression_results2 = symbolic_regression_on_bottleneck(X_query, X_key, np.expand_dims(Y2, axis=-1))


