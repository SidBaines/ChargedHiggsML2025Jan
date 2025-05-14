# %%
##############################################
##########       IMPORTS      ################
##############################################

######## Common imports ########
import numpy as np
import os
import torch
from datetime import datetime
import wandb
import shutil

######## My imports ########
from lowleveldataloader import ProportionalMemoryMappedDataset
from models import DeepSetsWithResidualSelfAttentionVariableTrueSkipClassifier
from lowlevelmetrics import HEPMetrics, HEPLoss, init_wandb
from utils import basic_lr_scheduler
from models import TestNetworkClassifier, GNNParticleClassifier, TestNetworkClassifierImmediatePool
from mechinterputils import run_with_cache_and_bottleneck



##############################################
#############       SETUP      ###############
##############################################
# %%
DRY_RUN=False # No saving plots, no wandb, no model saving, literally just to check a change hasn't broken the code
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

############   DATA PREP CONFIG  ################
# These are all variables which I need to specifcy exactly what the data prep was, 
#    so the model knows what form it's getting it in
ONLY_CORRECT_TRUTH_FOR_TRAINING = True
ONLY_CORRECT_RECO_FOR_TRAINING = True
REMOVE_WHERE_TRUTH_WOULD_BE_CUT=False # Should be false for classification train/test, and false for reco test
INCLUDE_INCLUSION_TAGS = True # This is only for newer files which contain these tags
# Want to re-do the prepdata and calculate the mH on the fly (using the truth reco) so we can put the correlation loss back in/trust the mH calculations
INCLUDE_ALL_SELECTIONS = True
INCLUDE_NEGATIVE_SELECTIONS = True
USE_OLD_TRUTH_SETTING = False
PHI_ROTATED = False
USE_LORENTZ_INVARIANT_FEATURES = True
TAG_INFO_INPUT = True
TOSS_UNCERTAIN_TRUTH = True
SHUFFLE_OBJECTS = True
NORMALISE_DATA = False
SCALE_DATA = True
CONVERT_TO_PT_PHI_ETA_M = False
IS_XBB_TAGGED = False
REQUIRE_XBB = False # If we only select categories 0, 3, 8, 9, 10 (ie, if INCLUDE_ALL_SELECTIONS is False) then I think this is satisfied anyway
INCLUDE_TAG_INFO = True
MET_CUT_ON = True
MH_SEL = False
USE_DROPOUT = True
if True:
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: # For testing new model architecture classes, probably run on cpu, since CUDA will just give difficult errors if there is some problem with the arch
    device = "cpu"
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
assert(not (NORMALISE_DATA and SCALE_DATA))
assert (~((REQUIRE_XBB and (~IS_XBB_TAGGED))))
if (not INCLUDE_ALL_SELECTIONS) and INCLUDE_NEGATIVE_SELECTIONS:
    assert(False)
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
if IS_XBB_TAGGED:
    N_CTX = 7 # the SIX types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
else:
    N_CTX = 6 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root->Binary files we're reading in.
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet'}
BIN_WRITE_TYPE=np.float32
max_n_objs_in_file = 30 # BE CAREFUL because this might change and if it does you ahve to rebinarise
max_n_objs_to_read = 14

if INCLUDE_TAG_INFO:
    N_Real_Vars_In_File = 5
    N_Real_Vars=5 # px, py, pz, energy, tagInfo.  BE CAREFUL because this might change and if it does you ahve to rebinarise
else:
    N_Real_Vars_In_File = 4
    N_Real_Vars=4 # px, py, pz, energy.  BE CAREFUL because this might change and if it does you ahve to rebinarise
if INCLUDE_INCLUSION_TAGS:
    N_Real_Vars_In_File += 2
    N_Real_Vars += 0


############   DATA LOADING CONFIG  ################
batch_size = 256
target_channel = 'lvbb'
validation_split_idx=0
n_splits=2
KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None


############   MODEL TRAINING CONFIG  ################
MODEL_ARCH="DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP"
ATTENTION_OUTPUT_BOTTLENECK_SIZE = 1
USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION = True
num_blocks_variable=10
model_cfg = {'d_attn':None, 'include_mlp':True, 'd_model': 152, 'd_mlp': 400, 'num_blocks':num_blocks_variable, 'dropout_p': 0.0, "embedding_size":N_CTX, "num_heads":4}

num_epochs = 30
log_interval = int(50e3/batch_size)
longer_log_interval = 100000000000
SAVE_MODEL_EVERY = 5
name_mapping = {"DEEPSETS":"DS", 
                "HYBRID_SELFATTENTION_GATED":"DSSAGA", 
                "DEEPSETS_SELFATTENTION":"DSSA", 
                "DEEPSETS_SELFATTENTION_RESIDUAL":"DSSAR", 
                "DEEPSETS_SELFATTENTION_RESIDUAL_X2":"DSSAR2", 
                "DEEPSETS_SELFATTENTION_RESIDUAL_X3":"DSSAR3", 
                "DEEPSETS_RESIDUAL_VARIABLE_TRUESKIP":f"DSSARVTS{num_blocks_variable}",
                "PARTICLE_FLOW":"PF",
                "TRANSFORMER":"TF",
                }
config = {
        "learning_rate": 2e-4,
        "learning_rate_low": 1e-8,
        "learning_rate_log_decay":True,
        "architecture": "PhysicsTransformer",
        "dataset": "ATLAS_ChargedHiggs",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "wandb":True,
        "name":"_"+timeStr+"_LowLevel_"+name_mapping[MODEL_ARCH]+"_"+target_channel,
        "weight_decay":1e-10,
    }






# %%
##############################################
#########       DATA LOADING     #############
##############################################
DATA_PATH = '/data/atlas/baines/20250321v2_AppliedRecoNNSplit_WithEventNumbers_WithSmallRJetCloseToLJetRemovalDeltaRLT0.5' + '_RemovedWrongTruthForTraining'* ONLY_CORRECT_TRUTH_FOR_TRAINING + '_RemovedWrongRecoForTraining'*ONLY_CORRECT_RECO_FOR_TRAINING + '_NotPhiRotated'*(not PHI_ROTATED) + '_XbbTagged'*IS_XBB_TAGGED + '_WithRecoMasses_' + f'{max_n_objs_in_file}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '_WithTagInfo'*INCLUDE_TAG_INFO + '_KeepAllOldSel'*INCLUDE_ALL_SELECTIONS  + 'IncludingNegative'*INCLUDE_NEGATIVE_SELECTIONS + '_RemovedEventsWhereTruthIsCutByMaxObjs'*REMOVE_WHERE_TRUTH_WOULD_BE_CUT +'/'
# assert(validation_split_idx<n_splits)
target_channel_num = {'lvbb':1, 'qqbb':2}[target_channel]
if 'AppliedRecoNN' in DATA_PATH: # This will have an extra variable, for the reco networks selection
    N_Real_Vars_In_File += 1
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

# %%
####### Get list of relevant input files ########
memmap_paths_train = {}
memmap_paths_val = {}
for file_name in os.listdir(DATA_PATH):
    if ('shape' in file_name) or ('npy' in file_name):
        continue
    if not (target_channel in file_name):
        continue
    dsid = file_name[5:11]
    if (int(dsid) > 500000) and (int(dsid) < 600000):
        if KEEP_DSID is not None:
            if (int(dsid)==KEEP_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MIN_DSID is not None: # Train with only certain DSID or above
            if (int(dsid)>=MIN_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MAX_DSID is not None: # Train with only certain DSID or below
            if (int(dsid)<=MAX_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        else: # train with all
            memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    else: # Keep all the background
        memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    memmap_paths_val[int(dsid)] = DATA_PATH+file_name

####### Create the dataloaders ########
train_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_train,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars_In_File,
                 N_Real_Vars_To_Return=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                #  shuffle_batch=False,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 has_eventNumbers=True,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths_val,  # DSID to memmap path
                 max_objs_in_memmap=max_n_objs_in_file,
                 N_Real_Vars_In_File=N_Real_Vars_In_File,
                 N_Real_Vars_To_Return=N_Real_Vars,
                 class_proportions = None,
                #  batch_size=64*8*64*8,
                 batch_size=batch_size,
                 device=device,
                 is_train=False,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                #  shuffle_batch=False,
                 means=means,
                 stds=stds,
                 objs_to_output=max_n_objs_to_read,
                 has_eventNumbers=True,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
print(train_dataloader.get_total_samples())
print(val_dataloader.get_total_samples())
# assert(False) # Need to check if the weighting is correct - it seemed likely that the sum of training weights for signal was not the same as for background?


# %%
##############################################
########       CREATE MODEL      #############
##############################################
# model = DeepSetsWithResidualSelfAttentionVariableTrueSkipClassifier(input_dim=N_Real_Vars, hidden_dim_mlp=model_cfg['d_mlp'], include_mlp=model_cfg['include_mlp'], num_attention_blocks=model_cfg['num_blocks'], hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
model = TestNetworkClassifier(hidden_dim_attn=model_cfg['d_attn'], use_lorentz_invariant_features=USE_LORENTZ_INVARIANT_FEATURES, bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, num_particle_types=N_CTX, hidden_dim_mlp=model_cfg['d_mlp'], include_mlp=model_cfg['include_mlp'], num_attention_blocks=model_cfg['num_blocks'], hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
# model = TestNetworkClassifierImmediatePool(hidden_dim_attn=model_cfg['d_attn'], use_lorentz_invariant_features=USE_LORENTZ_INVARIANT_FEATURES, bottleneck_attention=ATTENTION_OUTPUT_BOTTLENECK_SIZE, feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, num_particle_types=N_CTX, hidden_dim_mlp=model_cfg['d_mlp'], include_mlp=model_cfg['include_mlp'], num_attention_blocks=model_cfg['num_blocks'], hidden_dim=model_cfg['d_model'],  dropout_p=model_cfg['dropout_p'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
# model = GNNParticleClassifier(conv_type='edge', feature_set=['phi', 'eta', 'pt', 'm']+['tag']*TAG_INFO_INPUT, num_particle_types=N_CTX, num_layers=model_cfg['num_blocks'], hidden_dim=model_cfg['d_model'],  num_heads=model_cfg['num_heads'], embedding_size=model_cfg['embedding_size']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=config['weight_decay'])
print(sum(p.numel() for p in model.parameters()))
for p in model.named_parameters():
    print(f"{p[0]:50s}: {p[1].numel():10d}")

# %%
if DRY_RUN:
    config["wandb"] = False
if config['wandb']:
    init_wandb(config)
    # wandb.watch(model, log_freq=100)

# %%
#####################################################
#####     OPTIONALLY LOAD PRETRAINED MODEL     ######
#####################################################
if 0: # 
    print("WARNING: You are starting from a semi-pre-trained model state")
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250207-164605_TrainingOutput/models/0/chkpt49_357100.pth" # d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
    model.load_state_dict(loaded_state_dict)




# %%
##############################################
#######   SET UP METRIC TRACKERS      ########
##############################################
have_saved_a_model = False
if not USE_ENTROPY_TO_ENCOURAGE_SIMPLEATTENTION:
    criterion = HEPLoss(N_CTX, apply_correlation_penalty=False, alpha=1.0, calc_entropy_loss=False)
else:
    criterion = HEPLoss(N_CTX, apply_correlation_penalty=False, alpha=1.0, use_entropy_loss=True, entropy_weight=1e-3)

train_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
train_metrics_MCWts = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics_MCWts = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
global_step = 0
total_train_samples_processed = 0
train_dataloader._reset_indices()
orig_len_train_dataloader=len(train_dataloader)
orig_len_train_dataloader = int(orig_len_train_dataloader*0.75) # Only go through 0.6 of the data, since if we go all the way through, we'll just be running on background for the last few. There should be a better way of doing this, but I don't have it at present...
num_lr_steps = num_epochs*orig_len_train_dataloader



##############################################
##########   RUN TRAINING LOOP      ##########
##############################################
for epoch in range(num_epochs):
    ######## Reset dataloaders & metrics ########
    train_dataloader._reset_indices()
    val_dataloader._reset_indices()
    train_metrics.reset()
    train_metrics_MCWts.reset()
    val_metrics.reset()
    val_metrics_MCWts.reset()
    model.train()
    n_step = 0
    train_loss_epoch = 0
    sum_weights_epoch = 0
    for batch_idx in range(orig_len_train_dataloader):
        ######## Calculate learning rate ########
        if ((batch_idx%10)==0):
            new_lr = basic_lr_scheduler(batch_idx + epoch*orig_len_train_dataloader, config['learning_rate'], config['learning_rate_low'], num_lr_steps, config["learning_rate_log_decay"]) # Or in theory could use global step
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            if ((batch_idx%1000)==0):
                print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {new_lr:.6e}")
        n_step+=1
        global_step += 1

        ######## Training forward pass  ########
        batch = next(train_dataloader)
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        optimizer.zero_grad()
        if (ATTENTION_OUTPUT_BOTTLENECK_SIZE is None) and (not criterion.calc_entropy_loss):
            outputs = model(x, types)
        else:
            outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
        outputs[:, 3-target_channel_num] = -100 # Have to zero out the other channel, since we are only training on one channel at a time
        if (not criterion.calc_entropy_loss):
            loss = criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs)
        else:
            loss = criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs, cache=cache)
        train_loss_epoch += loss.item() * w.sum().item()
        sum_weights_epoch += w.sum().item()
        ######## Training backward pass  ########
        loss.backward()
        ######## Training gardient step  ########
        optimizer.step()

        ######## Update training metrics  ########
        total_train_samples_processed += len(y)
        train_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv, dsids, mHs)
        train_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
        if (n_step % 10) == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, orig_len_train_dataloader, loss.item()))
        
        ######## (Maybe) Log training metrics  ########
        if ((batch_idx % log_interval) == (log_interval-1)):
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if config['wandb']:
                wandb.log({"train/lr": current_lr}, commit=False)
                wandb.log({"train_samps_processed": total_train_samples_processed}, commit=False)
            log_level = 2 if ((batch_idx % (longer_log_interval)) == (longer_log_interval-1)) else 0
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=log_level, save=config['wandb'], commit=False)
            train_metrics_MCWts.compute_and_log(epoch, prefix="train_MC", step=global_step, log_level=log_level, save=config['wandb'], commit=True)
            train_metrics.reset_starts(ks=['sig_sel'])
            train_metrics_MCWts.reset_starts(ks=['sig_sel'])
    
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

    ######## Validation loader loop  ########
    if (epoch % 1) == 0:
        # Validation phase
        model.eval()
        with torch.no_grad():
            orig_len_val_dataloader=len(val_dataloader)
            loss = 0
            wt_sum = 0
            for batch_idx in range(orig_len_val_dataloader):
                batch = next(val_dataloader)
                # if (batch_idx >= orig_len_val_dataloader-5):
                #     continue

                x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                # x, y, w, types, mqq, mlv, MCWts = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
                if (ATTENTION_OUTPUT_BOTTLENECK_SIZE is None) and (not criterion.calc_entropy_loss):
                    outputs = model(x, types)
                else:
                    outputs, cache = run_with_cache_and_bottleneck(model, x[...,:4+int(TAG_INFO_INPUT)], types, detach=False)
                if 0:
                    loss += criterion(outputs[:,[0, target_channel_num]], y[:,[0, target_channel_num]], w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                    wt_sum += w.sum()
                    outputs[:, 3-target_channel_num] = -100
                else:
                    outputs[:, 3-target_channel_num] = -100
                    if (not criterion.calc_entropy_loss):
                        loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs).sum() * w.sum()
                    else:
                        loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs, cache=cache).sum() * w.sum()
                    wt_sum += w.sum()
                val_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv, dsids, mHs)
                val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
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
        
        ######## Log validation metrics  ########
        if config['wandb']:
            log_level = 3
            val_metrics.compute_and_log(epoch, prefix="val", log_level=log_level, save=config['wandb'], commit=False)
            val_metrics_MCWts.compute_and_log(epoch, prefix="val_MC", log_level=log_level, save=config['wandb'], commit=True)
    
    ######## (Maybe) save model  ########
    if ((epoch % SAVE_MODEL_EVERY) == (SAVE_MODEL_EVERY-1)):
        try:
            modelSaveDir = "%s/models/%s_Nplits%d_ValIdx%d/"%(saveDir, target_channel, n_splits, validation_split_idx)
            os.makedirs(modelSaveDir, exist_ok=True)
            torch.save(model.state_dict(), modelSaveDir + "/chkpt%d_%d" %(epoch, global_step) + '.pth')
            have_saved_a_model = True
        except:
            pass
# wandb.finish()

# %%
##############################################
###########   SAVE FINAL MODEL      ##########
##############################################
if not have_saved_a_model:
    modelSaveDir = "%s/models/%s_Nplits%d_ValIdx%d/"%(saveDir, target_channel, n_splits, validation_split_idx)
    os.makedirs(modelSaveDir, exist_ok=True)
    torch.save(model.state_dict(), modelSaveDir + "/chkpt%d" %(global_step) + '.pth')
