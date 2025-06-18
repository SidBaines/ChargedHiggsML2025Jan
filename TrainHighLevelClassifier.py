# %% # Load required modules
import time
ts = []
ts.append(time.time())

import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import torch
from datetime import datetime
from jaxtyping import Float
import einops
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg') # If you want to run in batch mode, and not see the plots made
# from utils import decode_y_eval_to_info
from dataloaders.highleveldataloader import ProportionalMemoryMappedDatasetHighLevel
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
import wandb
import torch.nn.functional as F
from models.models import ConfigurableNN
# from torchmetrics import Accuracy, AUC, ConfusionMatrix
# from torchmetrics import ConfusionMatrix
# from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import shutil
from metrics.highlevelmetrics import HEPMetrics, HEPLoss, init_wandb
from typing import List
from utils.utils import DSID_MASS_MAPPING
sorted_masses = sorted(list(DSID_MASS_MAPPING.values()))

# %%
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")
saveDir = "output/" + timeStr  + "_TrainingOutput/"
os.makedirs(saveDir)
print(saveDir)
if 0:
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
# Save the current script to the saveDir so we know what the training script was
def save_current_script(destination_directory):
    current_script_path = os.path.abspath(__file__)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    destination_path = os.path.join(destination_directory, os.path.basename(current_script_path))
    shutil.copy(current_script_path, destination_path)
    print(f"This script copied to: {destination_path}")
save_current_script('%s'%(saveDir))

# Some choices about the training process
# Assumes that the data has already been binarised
PARAMETRISED_NN = False
TOSS_UNCERTAIN_TRUTH = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = True
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here (well, really in the prep data script) or not
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 2 # Number of target classes (needed for one-hot encoding)
BIN_WRITE_TYPE=np.float32
N_Real_Vars = 7 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Set up stuff to read in data from bin file

batch_size = 64*2
DATA_PATH = '/data/atlas/baines/20250322v4_highLevel' + '_MetCut'*MET_CUT_ON + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '/'
# DATA_PATH = '/data/atlas/baines/20250429v1_HighLevelAppliedRecoNNSplit/' # For data which was recostructed using low-level network then the relevant high-level data was calculated
memmap_paths_train = {}
memmap_paths_val = {}
channel = 'qqbb'
n_splits=2
validation_split_idx=0
KEEP_DSID= None # a dsid if we only want to keep that DSID in training, or None
MIN_DSID = None # a dsid if we only want to keep this DSID or above (inclusive) or None
MAX_DSID = None # a dsid if we only want to keep this DSID or below (inclusive) or None
EXCL_DSID = None # a dsid if we only want to exclude that DSID in training, or None
assert(sum([i is not None for i in [KEEP_DSID, MIN_DSID, MAX_DSID, EXCL_DSID]])<=1)
assert(not (PARAMETRISED_NN and (KEEP_DSID is not None)))
means = np.load(f'{DATA_PATH}{channel}_mean.npy')
stds = np.load(f'{DATA_PATH}{channel}_std.npy')
for file_name in os.listdir(DATA_PATH):
    # if not '510124' in file_name:
    #     continue
    if 'shape' in file_name:
        continue
    if (channel in file_name) and ('dsid' in file_name):
        dsid = file_name[5:11]
        memmap_paths_val[int(dsid)] = DATA_PATH+file_name
        # if (int(dsid) > 510116) or (int(dsid) < 500000) or (int(dsid) > 600000):
        if KEEP_DSID is not None:
            if (int(dsid)==KEEP_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MIN_DSID is not None: # Train with only certain DSID or above
            if (int(dsid)>=MIN_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif MAX_DSID is not None: # Train with only certain DSID or below
            if (int(dsid)<=MAX_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        elif EXCL_DSID is not None:
            if (int(dsid)!=EXCL_DSID) or (int(dsid) < 500000) or (int(dsid) > 600000):
                memmap_paths_train[int(dsid)] = DATA_PATH+file_name
        else: # train with all
            memmap_paths_train[int(dsid)] = DATA_PATH+file_name
    else:
        pass
train_dataloader = ProportionalMemoryMappedDatasetHighLevel(
                 memmap_paths = memmap_paths_train,  # DSID to memmap path
                 N_Real_Vars=N_Real_Vars, # Will auto add 6 (for the y, w, dsid, mWh, mH, eventNumbers) inside the funciton
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 means=means,
                 stds=stds,
                 has_eventNumbers=True,
                 return_pole_mass=PARAMETRISED_NN,
                #  signal_reweights=np.array([3,3,3,1,1,1,1,1,1,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader = ProportionalMemoryMappedDatasetHighLevel(
                 memmap_paths = memmap_paths_val,  # DSID to memmap path
                 N_Real_Vars=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 validation_split_idx=validation_split_idx,
                 n_splits=n_splits,
                 n_targets=N_TARGETS,
                 means=means,
                 stds=stds,
                 has_eventNumbers=True,
                 return_pole_mass=PARAMETRISED_NN,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
# batch = next(train_dataloader)

# %%
print(train_dataloader.get_total_samples())
batch = next(train_dataloader)
# plt.hist(batch['mWh'].detach().cpu())

# %%
# Create a new model
models = {}
fit_histories = {}
model_n = 0



# models[model_n] = {'model' : Net(model_cfg).to(device), 'inputs' : inputs}
# models[model_n] = {'model' : MyHookedTransformer(model_cfg).to(device)}

models[model_n] = {
    'model':ConfigurableNN(
        N_inputs=N_Real_Vars+int(PARAMETRISED_NN), 
        N_targets=N_TARGETS, 
        # hidden_layers=[400,800,800,400,400],
        # hidden_layers=[128, 128, 128],
        hidden_layers=[256, 256, ],
        dropout_prob=0.0,
        use_batchnorm=False
        ).to(device)
    }
models[model_n]['model'].summary()

# %%
class_weights = [1 for _ in range(N_TARGETS)]
labels = ['Bkg', 'Lep', 'Had'] # truth==0 is bkg, truth==1 is leptonic decay, truth==2 is hadronic decay
class_weights_expanded = einops.repeat(torch.Tensor(class_weights), 't -> batch t', batch=batch_size).to(device)
# Cosine learning rate scheduler parameters
from utils.utils import basic_lr_scheduler


# SHOULD CHANGE WEIGHT DECAY BACK (IT WAS 1e-5 before)
num_epochs = 50

# %%
model, train_loader, val_loader = models[model_n]['model'], train_dataloader, val_dataloader
log_interval = 50
longer_log_interval = 1000000000
SAVE_MODEL_EVERY = int(num_epochs/3)
config = {
        "learning_rate_high": 5e-5,
        "learning_rate_low": 2e-7,
        # "learning_rates": [1e-4, 1e-5, 1e-6],
        "cosine_lr_n_epochs": num_epochs,   # Number of epochs to complete one cycle of learning rate
        "architecture": "HighLevelNN",
        "dataset": "ATLAS_ChargedHiggs",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "wandb":True,
        # "wandb":False,
        "name":"_"+timeStr+"_HighLevel"+channel+"_Parametrised"*PARAMETRISED_NN+f"_Only{str(KEEP_DSID)}"*(KEEP_DSID is not None)+f"_Min{str(MIN_DSID)}"*(MIN_DSID is not None)+f"_Max{str(MAX_DSID)}"*(MAX_DSID is not None)+f"_Excl{str(EXCL_DSID)}"*(EXCL_DSID is not None),
        # "weight_decay":2e-5,
        "weight_decay":1e-10,
    }
optimizer = torch.optim.Adam(models[model_n]['model'].parameters(), lr=1e-4, weight_decay=config["weight_decay"])
if config['wandb']:
    init_wandb(config)
    # wandb.watch(model, log_freq=100)

criterion = HEPLoss(apply_correlation_penalty=False, alpha=1.0)
train_metrics = HEPMetrics(parametrised_nn=PARAMETRISED_NN, max_bkg_levels=[200], max_buffer_len=int(train_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[1000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics = HEPMetrics(parametrised_nn=PARAMETRISED_NN, max_bkg_levels=[200], max_buffer_len=int(val_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=val_dataloader.abs_weight_sums, signal_acceptance_levels=[1000])
train_metrics_MCWts = HEPMetrics(parametrised_nn=PARAMETRISED_NN, max_bkg_levels=[200], max_buffer_len=int(train_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[1000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics_MCWts = HEPMetrics(parametrised_nn=PARAMETRISED_NN, max_bkg_levels=[200], max_buffer_len=int(val_dataloader.get_total_samples()), channel=channel, total_weights_per_dsid=val_dataloader.weight_sums, signal_acceptance_levels=[1000])
global_step = 0
total_train_samples_processed = 0

# %%
for epoch in range(num_epochs):
    # Update learning rate based on the cosine scheduler
    # new_lr = cosine_lr_scheduler(epoch, config['learning_rate'], config['learning_rate_low'], config['cosine_lr_n_epochs'])
    new_lr = basic_lr_scheduler(epoch, config['learning_rate_high'], config['learning_rate_low'], num_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {new_lr:.6e}")
    # Training phase
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
        n_step+=1
        global_step += 1
        if (batch_idx >= orig_len_train_dataloader-5):
            continue
        batch = next(train_loader)
        if PARAMETRISED_NN:
            x, y, mWh, dsid, w, MC_w, mH, pole_mass = batch.values()
        else:
            x, y, mWh, dsid, w, MC_w, mH = batch.values()
        # batch['train_wts']
        # assert(False)
        # x, y, w, mWh, dsid = x.to(device), y.to(device), w.to(device), mWh.to(device), dsid.to(device)
        
        optimizer.zero_grad()
        if PARAMETRISED_NN:
            outputs = model(torch.cat([x, pole_mass.unsqueeze(-1)], dim=-1))
        else:
            outputs = model(x)
        # assert(False)
        loss = criterion(outputs, y, w, config['wandb'], mWh, mH)
        train_loss_epoch += loss.item() * w.sum().item()
        sum_weights_epoch += w.sum().item()
        
        loss.backward()
        optimizer.step()
        
        # Update training metrics
        total_train_samples_processed += len(y)
        if PARAMETRISED_NN:
            # Need to re-do the outputs with each of the different masses as input
            outputs = torch.zeros(outputs.shape[0], 10, outputs.shape[1]).to(device)
            for i in range(10):
                outputs[:, i, :] = model(torch.cat([x, torch.ones_like(pole_mass).unsqueeze(-1)*sorted_masses[i]], dim=-1))
        else:
            pass
        train_metrics.update(outputs, y.argmax(dim=-1), w, mWh, dsid, mH)
        train_metrics_MCWts.update(outputs, y.argmax(dim=-1), MC_w, mWh, dsid, mH)
        print_every_steps = 100
        if (n_step % print_every_steps) == (print_every_steps-1):
            print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, orig_len_train_dataloader, loss.item()))
            # Log training metrics every log_interval batches
        if ((batch_idx % log_interval) == 0):
            current_lr = optimizer.param_groups[0]['lr']
            if config['wandb']:
                wandb.log({"train/lr": current_lr}, commit=False)
                wandb.log({"train_samps_processed": total_train_samples_processed}, commit=False)
            log_level = 2 if ((batch_idx % longer_log_interval) == (longer_log_interval-1)) else 0
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=log_level, save=config['wandb'], commit=False)
            train_metrics_MCWts.compute_and_log(epoch, prefix="train_MC", step=global_step, log_level=log_level, save=config['wandb'], commit=True)
            train_metrics.reset_starts(ks=['sig_sel', 'auc'])
            train_metrics_MCWts.reset_starts(ks=['sig_sel', 'auc'])
            # Log learning rate
        
    if config['wandb']:
        log_level = 3
        wandb.log({'train/loss_total':train_loss_epoch/sum_weights_epoch}, commit=False)
        train_metrics.reset_starts()
        train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=log_level, save=config['wandb'], commit=False)
        train_metrics_MCWts.reset_starts()
        train_metrics_MCWts.compute_and_log(epoch, prefix="train_MC", step=global_step, log_level=log_level, save=config['wandb'], commit=True)
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"train/lr": current_lr}, step=global_step)
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
                }, step=global_step)
    
    if (epoch % 1) == 0:
        # Validation phase
        model.eval()
        with torch.no_grad():
            orig_len_val_dataloader=len(val_loader)
            loss = 0
            wt_sum = 0
            for batch_idx in range(orig_len_val_dataloader):
                if (batch_idx >= orig_len_val_dataloader-5):
                    continue
                batch = next(val_loader)

                if PARAMETRISED_NN:
                    x, y, mWh, dsid, w, MC_w, mH, pole_mass = batch.values()
                else:
                    x, y, mWh, dsid, w, MC_w, mH = batch.values()
                # x, y, w, mWh, dsid = x.to(device), y.to(device), w.to(device), mWh.to(device), dsid.to(device)
            
                if PARAMETRISED_NN:
                    outputs = model(torch.cat([x, pole_mass.unsqueeze(-1)], dim=-1))
                else:
                    outputs = model(x)

                loss += criterion(outputs, y, w, config['wandb'], mWh, mH).sum() * w.sum()
                wt_sum += w.sum()
                # Now calculate outputs again for each mass point for metrics
                if PARAMETRISED_NN:
                    outputs = torch.zeros(outputs.shape[0], 10, outputs.shape[1]).to(device)
                    for i in range(10):
                        outputs[:, i, :] = model(torch.cat([x, torch.ones_like(pole_mass).unsqueeze(-1)*sorted_masses[i]], dim=-1))
                else:
                    pass
                val_metrics.update(outputs, y.argmax(dim=-1), w, mWh, dsid, mH)
                val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MC_w, mWh, dsid, mH)
                # print('[%d/%d][%d/%d] Val' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader))
            if config['wandb']:
                wandb.log({"train_samps_processed": total_train_samples_processed}, commit=False)
                wandb.log({
                    "val/loss_total": loss.item(),
                    "val/loss_ce": loss.item()/wt_sum.item(),
                    # "loss/qq_mass": qq_mass_loss.item(),
                    # "loss/lv_mass": lv_mass_loss.item()
                })
            print('[%d/%d][%d/%d]\tVAL Loss_C: %.4e' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader, loss.item()/wt_sum.item()))
        
        # Log validation metrics
        if config['wandb']:
            log_level = 3
            val_metrics.compute_and_log(epoch, prefix="val", step=global_step, log_level=log_level, save=config['wandb'], commit=False)
            val_metrics_MCWts.compute_and_log(epoch, prefix="val_MC", step=global_step, log_level=log_level, save=config['wandb'], commit=True)

    
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
            modelSaveDir = "%s/models/%s_Nplits%d_ValIdx%d/"%(saveDir, channel, n_splits, validation_split_idx)
            os.makedirs(modelSaveDir, exist_ok=True)
            torch.save(models[model_n]["model"].state_dict(), modelSaveDir + "/chkpt%d" %(global_step) + '.pth')
            shutil.copy(f'{DATA_PATH}{channel}_std.npy', modelSaveDir + "/std.npy")
            shutil.copy(f'{DATA_PATH}{channel}_mean.npy', modelSaveDir + "/mean.npy")
        except:
            pass
wandb.finish()

# %%

modelSaveDir = "%s/models/%s_Nplits%d_ValIdx%d/"%(saveDir, channel, n_splits, validation_split_idx)
os.makedirs(modelSaveDir, exist_ok=True)
torch.save(models[model_n]["model"].state_dict(), modelSaveDir + "/chkpt%d" %(global_step) + '.pth')
shutil.copy(f'{DATA_PATH}{channel}_std.npy', modelSaveDir + "/std.npy")
shutil.copy(f'{DATA_PATH}{channel}_mean.npy', modelSaveDir + "/mean.npy")
# %%
