# %% # Load required modules
import time
ts = []
ts.append(time.time())

import numpy as np
import os
import torch
from datetime import datetime
from jaxtyping import Float
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
from MyMetricsLowLevel import HEPMetrics, HEPLoss, init_wandb

# %%
timeStr = datetime.now().strftime("%Y%m%d-%H%M%S")
saveDir = "output/" + timeStr  + "_TrainingOutput/"
os.makedirs(saveDir)
print(saveDir)
if 1:
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
TOSS_UNCERTAIN_TRUTH = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = True
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here (well, really in the prep data script) or not
SHUFFLE_OBJECTS = True
NORMALISE_DATA = False
CONVERT_TO_PT_PHI_ETA_M = False
MET_CUT_ON = True
MH_SEL = False
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
N_CTX = 7 # the six types of object, plus one for 'no object;. We need to hardcode this unfortunately
BIN_WRITE_TYPE=np.float32
max_n_objs = 14 # BE CAREFUL because this might change and if it does you ahve to rebinarise
N_Real_Vars = 4 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise
types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}

# %%
# Set up stuff to read in data from bin file

batch_size = 64*8
DATA_PATH=f'/data/atlas/baines/tmp2_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired' + '_mHSel'*MH_SEL + '/'
DATA_PATH=f'/data/atlas/baines/tmp_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired' + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '/'
if NORMALISE_DATA:
    means = np.load(f'{DATA_PATH}mean.npy')[1:]
    stds = np.load(f'{DATA_PATH}std.npy')[1:]
else:
    means = None
    stds = None
memmap_paths = {}
for file_name in os.listdir(DATA_PATH):
    if ('shape' in file_name) or ('npy' in file_name):
        continue
    dsid = file_name[5:11]
    memmap_paths[int(dsid)] = DATA_PATH+file_name
train_split=0.5
train_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths,  # DSID to memmap path
                 max_n_objs=max_n_objs,
                 N_Real_Vars=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=True,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 train_split=train_split,
                 means=means,
                 stds=stds,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
val_dataloader = ProportionalMemoryMappedDataset(
                 memmap_paths = memmap_paths,  # DSID to memmap path
                 max_n_objs=max_n_objs,
                 N_Real_Vars=N_Real_Vars,
                 class_proportions = None,
                 batch_size=batch_size,
                 device=device, 
                 is_train=False,
                 n_targets=N_TARGETS,
                 shuffle=SHUFFLE_OBJECTS,
                 train_split=train_split,
                 means=means,
                 stds=stds,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
print(train_dataloader.get_total_samples())
# assert(False) # Need to check if the weighting is correct - it seemed likely that the sum of training weights for signal was not the same as for background?
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

class MyHookedTransformer(HookedTransformer):
    def __init__(self, cfg, mass_input_layer=2, mass_hidden_dim=256, **kwargs):
        super(MyHookedTransformer, self).__init__(cfg, **kwargs)
        self.hook_dict['hook_mytokens'] = HookPoint()
        self.hook_dict['hook_mytokens'].name = 'hook_mytokens'
        self.mod_dict['hook_mytokens'] = self.hook_dict['hook_mytokens']
        self.W_Embed = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_Embed, std=0.02)
        
    def forward(self, tokens: Float[Tensor, "batch object d_input"], token_types: Float[Tensor, "batch object"], **kwargs) -> Float[Tensor, "batch d_model"]:
        self.hook_dict['hook_mytokens'](tokens)
        expanded_W_E = self.W_Embed.unsqueeze(0).expand(token_types.shape[0], -1, -1, -1)
        expanded_types = token_types.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.W_Embed.shape[-2], self.W_Embed.shape[-1])
        W_E_selected = torch.gather(expanded_W_E, dim=1, index=expanded_types)
        output = einops.einsum(tokens, W_E_selected, "batch object d_input, batch object d_input d_model -> batch object d_model")
        if 'start_at_layer' in kwargs:
            raise NotImplementedError
        else:
            class_outs = super(MyHookedTransformer, self).forward(output, start_at_layer=0, **kwargs)
            class_outs = class_outs[:,0]
        return class_outs

# %%
# Create a new model
models = {}
fit_histories = {}
model_n = 0

# Create the model with the desired properties
model_cfg = HookedTransformerConfig(
    # normalization_type='LN',
    normalization_type='LN',
    d_model=16,
    d_head=8,
    n_layers=8,
    n_heads=8,
    n_ctx=N_CTX, # Max number of types of object per event + 1 because we want a dummy row in the embedding matrix for non-existing particles
    d_vocab=N_Real_Vars, # Number of inputs per object
    d_vocab_out=N_TARGETS,  # 2 because we're doing binary classification
    d_mlp=64,
    attention_dir="bidirectional",  # defaults to "causal"
    act_fn="relu",
    use_attn_result=True,
    device=str(device),
    use_hook_tokens=True,
)

# models[model_n] = {'model' : Net(model_cfg).to(device), 'inputs' : inputs}
models[model_n] = {'model' : MyHookedTransformer(model_cfg).to(device)}
print(sum(p.numel() for p in models[model_n]['model'].parameters()))

# %%
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
        attn_scores.masked_fill_(model.hook_dict["hook_mytokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_mytokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_mytokens":
            hook.add_perma_hook(cache_padding_tokens_mask)  # type: ignore
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)  # type: ignore

    return model


models[model_n]['model'].reset_hooks(including_permanent=True)
models[model_n]['model'] = add_perma_hooks_to_mask_pad_tokens(models[model_n]['model'])
class_weights = [1 for _ in range(N_TARGETS)]
labels = ['Bkg', 'Lep', 'Had'] # truth==0 is bkg, truth==1 is leptonic decay, truth==2 is hadronic decay
class_weights_expanded = einops.repeat(torch.Tensor(class_weights), 't -> batch t', batch=batch_size).to(device)

# %%

# Cosine learning rate scheduler parameters
import math
def cosine_lr_scheduler(epoch: int, lr_high: float, lr_low: float, n_epochs: int):
    """
    This function calculates the learning rate following a cosine schedule
    that oscillates between `lr_high` and `lr_low` every `n_epochs`.
    """
    # Cosine annealing function
    cycle_epoch = epoch % n_epochs
    progress = cycle_epoch / n_epochs
    lr = lr_low + 0.5 * (lr_high - lr_low) * (1 + math.cos(math.pi * progress))
    return lr


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

# %%
model, train_loader, val_loader = models[model_n]['model'], train_dataloader, val_dataloader
num_epochs = 300
log_interval = 100
# longer_log_interval = log_interval*10
longer_log_interval = 100000000000
SAVE_MODEL_EVERY = 30
config = {
        "learning_rate": 5e-5,
        "learning_rate_low": 1e-5,
        "cosine_lr_n_epochs": num_epochs,   # Number of epochs to complete one cycle of learning rate
        "architecture": "PhysicsTransformer",
        "dataset": "ATLAS_ChargedHiggs",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "wandb":True,
        "name":"_"+timeStr+"_LowLevel",
        "weight_decay":1e-4,
    }
optimizer = torch.optim.Adam(models[model_n]['model'].parameters(), lr=1e-4, weight_decay=config['weight_decay'])
if config['wandb']:
    init_wandb(config)
    # wandb.watch(model, log_freq=100)

# %%
if 0: # 
    print("WARNING: You are starting from a semi-pre-trained model state")
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250131-112943_TrainingOutput/models/0/chkpt178200.pth" # d_model=128, d_head=8,    n_layers=8,    n_heads=4,    n_ctx=N_CTX,   d_vocab=N_Real_Vars,   d_vocab_out=N_TARGETS,  d_mlp=256,
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250130-153911_TrainingOutput/models/0/chkpt178200.pth" # d_model=64
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250131-163426_TrainingOutput/models/0/chkpt134000.pth" # d_model=128
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250201-122944_TrainingOutput/models/0/chkpt329640.pth" # d_model=128
    # modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250207-173524_TrainingOutput/models/0/chkpt49_357100.pth" # d_model=64,    d_head=8,    n_layers=4,    n_heads=4,    d_mlp=256,
    modelfile="/users/baines/Code/ChargedHiggs_ExperimentalML/output/20250207-164605_TrainingOutput/models/0/chkpt49_357100.pth" # d_model=32,    d_head=8,    n_layers=8,    n_heads=8,    d_mlp=128,
    loaded_state_dict = torch.load(modelfile, map_location=torch.device(device))
    models[model_n]['model'].load_state_dict(loaded_state_dict)
# %%
criterion = HEPLoss()
train_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.abs_weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
train_metrics_MCWts = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(train_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000]) # TODO should 'total_weights_per_dsid' here be abs or not-abs
val_metrics_MCWts = HEPMetrics(max_bkg_levels=[100, 200], max_buffer_len=int(val_dataloader.get_total_samples()), total_weights_per_dsid=train_dataloader.weight_sums, signal_acceptance_levels=[100, 500, 1000, 5000])
global_step = 0
for epoch in range(num_epochs):
    # Update learning rate based on the cosine scheduler
    new_lr = cosine_lr_scheduler(epoch, config['learning_rate'], config['learning_rate_low'], config['cosine_lr_n_epochs'])
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
        x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
        # x, y, w, types, mqq, mlv, MCWts, mH = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
        
        optimizer.zero_grad()
        outputs = model(x, types)
        loss = criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs)
        train_loss_epoch += loss.item()
        sum_weights_epoch += w.sum().item()
        
        loss.backward()
        optimizer.step()
        
        # Update training metrics
        train_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv, dsids, mHs)
        train_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
        if (n_step % 10) == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, orig_len_train_dataloader, loss.item()))
        # Log training metrics every log_interval batches
        if ((batch_idx % log_interval) == 0):
            log_level = 2 if ((batch_idx % (longer_log_interval)) == (longer_log_interval-1)) else 0
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=log_level, save=config['wandb'])
            train_metrics_MCWts.compute_and_log(epoch, prefix="train_MC", step=global_step, log_level=log_level, save=config['wandb'])
            train_metrics.reset_starts(ks=['sig_sel'])
            train_metrics_MCWts.reset_starts(ks=['sig_sel'])
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if config['wandb']:
                wandb.log({"train/lr": current_lr}, step=global_step)
    
    # if config['wandb']:
    if 1:
        if config['wandb']:
            wandb.log({'train/loss_total':train_loss_epoch/sum_weights_epoch}, commit=False)
        if (epoch % 1) == 0:
            log_level = 3
            train_metrics.reset_starts()
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=log_level, save=config['wandb'])
            train_metrics_MCWts.reset_starts()
            train_metrics_MCWts.compute_and_log(epoch, prefix="train_MC", step=global_step, log_level=log_level, save=config['wandb'])
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if config['wandb']:
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
                batch = next(val_loader)
                if (batch_idx >= orig_len_val_dataloader-5):
                    continue

                x, y, w, types, dsids, mqq, mlv, MCWts, mHs = batch.values()
                # x, y, w, types, mqq, mlv, MCWts = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device), MCWts.to(device)
                
                outputs = model(x, types)
                loss += criterion(outputs, y, w, config['wandb'], mqq, mlv, mHs).sum()
                wt_sum += w.sum()
                val_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv, dsids, mHs)
                val_metrics_MCWts.update(outputs, y.argmax(dim=-1), MCWts, mqq, mlv, dsids, mHs)
                # print('[%d/%d][%d/%d] Val' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader))
                if (n_step % 10) == 0:
                    print('[%d/%d][%d/%d]\tVAL Loss_C: %.4e' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader, loss.item()/wt_sum.item()))
            if config['wandb']:
                wandb.log({
                    "val/loss_total": loss.item()/wt_sum.item(),
                    "val/loss_ce": loss.item()/wt_sum.item(),
                    # "loss/qq_mass": qq_mass_loss.item(),
                    # "loss/lv_mass": lv_mass_loss.item()
                })
            # print('[%d/%d][%d/%d]\tVAL Loss_C: %.4e' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader, loss.item()/wt_sum.item()))
        
        # Log validation metrics
        if config['wandb']:
            log_level = 3
            val_metrics.compute_and_log(epoch, prefix="val", step=global_step, log_level=log_level, save=config['wandb'])
            val_metrics_MCWts.compute_and_log(epoch, prefix="val_MC", step=global_step, log_level=log_level, save=config['wandb'])

    
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
            modelSaveDir = "%s/models/%d/"%(saveDir, model_n)
            os.makedirs(modelSaveDir, exist_ok=True)
            torch.save(models[model_n]["model"].state_dict(), modelSaveDir + "/chkpt%d_%d" %(epoch, global_step) + '.pth')
        except:
            pass
# wandb.finish()

# %%
modelSaveDir = "%s/models/%d/"%(saveDir, model_n)
os.makedirs(modelSaveDir, exist_ok=True)
torch.save(models[model_n]["model"].state_dict(), modelSaveDir + "/chkpt%d" %(global_step) + '.pth')
# %%
