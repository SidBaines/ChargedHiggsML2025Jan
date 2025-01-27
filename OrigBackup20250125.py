# %% [markdown]
#  # Load required modules

# %%
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
from utils import save_current_script, decode_y_eval_to_info
from mynewdataloader import ProportionalMemoryMappedDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
import wandb
import torch.nn.functional as F
# from torchmetrics import Accuracy, AUC, ConfusionMatrix
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# %%
saveDir = "output/" + datetime.now().strftime("%Y%m%d-%H%M%S")  + "_TrainingOutput/"
os.makedirs(saveDir)
print(saveDir)
if 1:
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
save_current_script('%s'%(saveDir))

# Some choices about the training process
# Assumes that the data has already been binarised
SHUFFLE_OBJECTS = False
CONVERT_TO_PT_PHI_ETA_M = False
MET_CUT_ON = True
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
N_CTX = 7 # the six types of object, plus one for 'no object;. We need to hardcode this unfortunately
BIN_WRITE_TYPE=np.float32
max_n_objs = 14 # BE CAREFUL because this might change and if it does you ahve to rebinarise
N_Real_Vars = 4 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise
types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Set up stuff to read in data from bin file

batch_size = 64*32
DATA_PATH=f'/data/atlas/baines/tmp_SingleXbbSelected_XbbTagged_WithRecoMasses_{max_n_objs}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '/'
memmap_paths = {}
for file_name in os.listdir(DATA_PATH):
    if 'shape' in file_name:
        continue
    dsid = file_name[5:11]
    memmap_paths[int(dsid)] = DATA_PATH+file_name
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
                 train_split=0.5,
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
                 train_split=0.95,
                #  signal_reweights=np.array([10,9,8,7,6,5,4,3,2,1]),
                #  signal_reweights=np.array([1e1, 1e1, 1e1, 1e0,1e0,1e0,1e-1,1e-1,1e-1,1e-2]),
)
# batch = next(train_dataloader)

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
    d_model=32,
    d_head=8,
    n_layers=4,
    n_heads=4,
    n_ctx=N_CTX, # Max number of types of object per event + 1 because we want a dummy row in the embedding matrix for non-existing particles
    d_vocab=N_Real_Vars, # Number of inputs per object
    d_vocab_out=N_TARGETS,  # 2 because we're doing binary classification
    d_mlp=128,
    attention_dir="bidirectional",  # defaults to "causal"
    act_fn="relu",
    use_attn_result=True,
    device=str(device),
    use_hook_tokens=True,
)

# models[model_n] = {'model' : Net(model_cfg).to(device), 'inputs' : inputs}
models[model_n] = {'model' : MyHookedTransformer(model_cfg).to(device)}

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
optimizer = torch.optim.Adam(models[model_n]['model'].parameters(), lr=1e-4)
num_epochs = 10
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

class SignalSelectionMetrics:
    def __init__(self, signal_acceptance_levels=[0.5, 0.7, 0.9]):
        """
        Args:
            signal_acceptance_levels: List of percentages (0-1) of signal events to accept
        """
        self.signal_levels = sorted(signal_acceptance_levels)
        self.reset()
        
    def reset(self):
        # Store all predictions and targets with weights
        self.all_probs = []
        self.all_targets = []
        self.all_weights = []
        
    def update(self, preds, targets, weights):
        """
        Args:
            preds: Tensor of shape [batch_size, 3] with predicted probabilities
                  (columns: background, lvbb, qqbb)
            targets: Tensor of shape [batch_size] with true class indices
            weights: Tensor of shape [batch_size] with sample weights
        """
        # Convert to probabilities if needed
        if preds.shape[1] > 3:  # If logits are passed
            probs = F.softmax(preds, dim=1)
        else:
            probs = preds
            
        self.all_probs.append(probs.cpu())
        self.all_targets.append(targets.cpu())
        self.all_weights.append(weights.cpu())
        
    def compute(self):
        if not self.all_probs:
            return {}
            
        # Concatenate all batches
        probs = torch.cat(self.all_probs)
        targets = torch.cat(self.all_targets)
        weights = torch.cat(self.all_weights)
        
        # Separate signal and background
        bkg_mask = targets == 0
        lvbb_mask = targets == 1
        qqbb_mask = targets == 2
        
        # Get probabilities
        p_bkg = probs[:, 0]
        p_lvbb = probs[:, 1]
        p_qqbb = probs[:, 2]
        
        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for each signal acceptance level
        for level in self.signal_levels:
            # Calculate thresholds for this acceptance level
            lvbb_thresh = self._find_threshold(
                p_bkg[lvbb_mask], p_lvbb[lvbb_mask], p_qqbb[lvbb_mask],
                weights[lvbb_mask], level
            )
            qqbb_thresh = self._find_threshold(
                p_bkg[qqbb_mask], p_qqbb[qqbb_mask], p_lvbb[qqbb_mask],
                weights[qqbb_mask], level
            )
            
            # Apply selection logic
            lvbb_selected = (
                (p_lvbb >= p_qqbb) & 
                (p_bkg < lvbb_thresh)
            )
            qqbb_selected = (
                (p_qqbb >= p_lvbb) & 
                (p_bkg < qqbb_thresh)
            )
            
            # Calculate background proportions
            bkg_lvbb = (bkg_mask & lvbb_selected).float() * weights
            bkg_qqbb = (bkg_mask & qqbb_selected).float() * weights
            
            total_bkg_weight = weights[bkg_mask].sum()
            
            if total_bkg_weight > 0:
                bkg_lvbb_frac = bkg_lvbb.sum() / total_bkg_weight
                bkg_qqbb_frac = bkg_qqbb.sum() / total_bkg_weight
            else:
                bkg_lvbb_frac = 0.0
                bkg_qqbb_frac = 0.0
                
            # Store results
            results[f'signal_acceptance_{int(level*100)}'] = {
                'lvbb_threshold': lvbb_thresh,
                'qqbb_threshold': qqbb_thresh,
                'bkg_lvbb_fraction': bkg_lvbb_frac.item(),
                'bkg_qqbb_fraction': bkg_qqbb_frac.item()
            }
            
        return results

    def _find_threshold(self, p_bkg, p_sig, p_other_sig, weights, acceptance_level, signal_type):
        """
        Find the background probability threshold that accepts the specified
        fraction of signal events, while respecting the mutual exclusivity rule.
        
        Args:
            p_bkg: Background probabilities for signal events
            p_sig: Signal probabilities (lvbb or qqbb) for signal events
            p_other_sig: Other signal probabilities (qqbb or lvbb) for signal events
            weights: Event weights
            acceptance_level: Desired fraction of signal to accept
            signal_type: 'lvbb' or 'qqbb' - determines which exclusivity rule to apply
        """
        if len(p_sig) == 0:
            return 1.0  # No signal events
        
        # Apply mutual exclusivity rule
        if signal_type == 'lvbb':
            valid_mask = p_sig >= p_other_sig
        else:  # qqbb
            valid_mask = p_sig >= p_other_sig
            
        # Filter events that satisfy the mutual exclusivity rule
        p_bkg = p_bkg[valid_mask]
        p_sig = p_sig[valid_mask]
        weights = weights[valid_mask]
        
        if len(p_sig) == 0:
            return 1.0  # No signal events after applying exclusivity
        
        # Sort signal events by p_bkg
        sorted_idx = torch.argsort(p_bkg)
        sorted_p_bkg = p_bkg[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Calculate cumulative sum of weights
        cum_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = cum_weights[-1]
        
        # Find threshold that accepts desired fraction
        target_weight = total_weight * acceptance_level
        idx = torch.searchsorted(cum_weights, target_weight)
        
        if idx >= len(sorted_p_bkg):
            return 1.0
        return sorted_p_bkg[idx].item()

class WeightedMulticlassAccuracy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.total_correct = 0.0
        self.total_weight = 0.0
        
    def update(self, preds, targets, weights=None):
        """
        Args:
            preds: Tensor of shape [batch_size, num_classes] with predicted logits
            targets: Tensor of shape [batch_size] with true class indices
            weights: Optional tensor of shape [batch_size] with sample weights
        """
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
            
        # Convert predictions to class indices
        pred_classes = torch.argmax(preds, dim=1)
        
        # Calculate weighted correct predictions
        correct = (pred_classes == targets).float()
        weighted_correct = correct * weights
        
        self.total_correct += weighted_correct.sum().item()
        self.total_weight += weights.sum().item()
        
    def compute(self):
        if self.total_weight == 0:
            return 0.0
        return self.total_correct / self.total_weight

class WeightedMulticlassAUC:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.all_probs = []
        self.all_targets = []
        self.all_weights = []
        
    def update(self, preds, targets, weights=None):
        """
        Args:
            preds: Tensor of shape [batch_size, num_classes] with predicted logits
            targets: Tensor of shape [batch_size] with true class indices
            weights: Optional tensor of shape [batch_size] with sample weights
        """
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
            
        probs = F.softmax(preds, dim=1)
        
        # Store for later computation
        self.all_probs.append(probs.cpu())
        self.all_targets.append(targets.cpu())
        self.all_weights.append(weights.cpu())
        
    def compute(self):
        if not self.all_probs:
            return 0.0
            
        # Concatenate all batches
        probs = torch.cat(self.all_probs)
        targets = torch.cat(self.all_targets)
        weights = torch.cat(self.all_weights)
        
        # Compute one-vs-rest AUC for each class
        auc_scores = []
        for class_idx in range(self.num_classes):
            # Create binary targets for this class
            binary_targets = (targets == class_idx).float()
            
            # Get probabilities for this class
            class_probs = probs[:, class_idx]
            
            # Compute weighted AUC
            if len(torch.unique(binary_targets)) < 2:
                # Only one class present, skip
                continue
                
            # Sort by predicted probability
            sort_idx = torch.argsort(class_probs)
            sorted_probs = class_probs[sort_idx]
            sorted_targets = binary_targets[sort_idx]
            sorted_weights = weights[sort_idx]
            
            # Compute weighted TPR and FPR
            total_pos_weight = (sorted_targets * sorted_weights).sum()
            total_neg_weight = ((1 - sorted_targets) * sorted_weights).sum()
            
            tpr = torch.cumsum(sorted_targets * sorted_weights, dim=0) / total_pos_weight
            fpr = torch.cumsum((1 - sorted_targets) * sorted_weights, dim=0) / total_neg_weight
            
            # Compute AUC using trapezoidal rule
            auc = torch.trapz(tpr, fpr).item()
            auc_scores.append(auc)
        
        if not auc_scores:
            return 0.0
        return sum(auc_scores) / len(auc_scores)

class WeightedConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.float32
        ).cpu()
        
    def update(self, preds, targets, weights=None):
        """
        Args:
            preds: Tensor of shape [batch_size, num_classes] with predicted logits
            targets: Tensor of shape [batch_size] with true class indices
            weights: Optional tensor of shape [batch_size] with sample weights
        """
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
            
        # Convert predictions to class indices
        pred_classes = torch.argmax(preds, dim=1)
        
        # Create one-hot encoding of predictions and targets
        pred_onehot = F.one_hot(pred_classes, num_classes=self.num_classes).float()
        target_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Outer product to get confusion matrix contributions
        batch_conf = torch.matmul(
            target_onehot.transpose(0, 1),  # [num_classes, batch_size]
            pred_onehot * weights.unsqueeze(-1)  # [batch_size, num_classes]
        )
        
        self.conf_matrix += batch_conf.cpu()
        
    def compute(self):
        return self.conf_matrix
    
    def plot(self, class_names=None):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
            class_names: List of class names for labeling axes
        """
        conf_matrix = self.conf_matrix.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(conf_matrix, cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Add labels
        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
            
        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Confusion Matrix",
            ylabel="True label",
            xlabel="Predicted label"
        )
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if conf_matrix.dtype.kind == 'f' else 'd'
        thresh = conf_matrix.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, format(conf_matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig

def init_wandb(config):
    wandb.init(
        project="HEP-Transformers",
        config=config,
        magic=True
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")


class HEPMetrics:
    def __init__(self, 
                 num_classes=3, 
                 mass_bins=50, 
                 mass_range=(0, 1000), 
                 signal_acceptance_levels=[0.5, 0.7, 0.9]
                 ):
        # Classification metrics
        self.accuracy = WeightedMulticlassAccuracy(num_classes=num_classes)
        self.auc = WeightedMulticlassAUC(num_classes=num_classes)
        self.confusion = WeightedConfusionMatrix(num_classes=num_classes)
        self.confusion = WeightedConfusionMatrix(num_classes=num_classes)
        self.signal_selection = SignalSelectionMetrics(signal_acceptance_levels)
        
        
        # Mass reconstruction histograms
        self.mass_bins = mass_bins
        self.mass_range = mass_range
        self.qqbb_mass = []
        self.lvbb_mass = []
        
        # Significance tracking
        self.true_signal = []
        self.pred_scores = []
        self.targets = []
    
    def update(self, preds, targets, weights, masses_qq, masses_lv):
        probs = F.softmax(preds, dim=1)
        self.accuracy.update(preds, targets, weights)
        # print("Accuracy calculated here: ", self.accuracy.compute())
        self.auc.update(preds, targets, weights)
        self.confusion.update(preds, targets, weights)
        self.signal_selection.update(preds, targets, weights)
        
        # Store masses for correctly classified signals
        sig_mask = targets > 0
        pred_labels = preds.argmax(dim=1)
        
        qq_correct = (targets == 2) & (pred_labels == 2)
        lv_correct = (targets == 1) & (pred_labels == 1)
        
        self.qqbb_mass.extend(masses_qq[qq_correct].cpu().tolist())
        self.lvbb_mass.extend(masses_lv[lv_correct].cpu().tolist())
        
        # For significance calculation
        self.true_signal.extend((targets > 0).cpu().tolist())
        self.pred_scores.extend(probs[:,1:].sum(dim=1).cpu().tolist())
        self.targets.extend(targets.cpu().tolist())
    
    def reset(self):
        # Reset metrics
        self.accuracy.reset()
        self.auc.reset()
        self.confusion.reset()
        self.qqbb_mass = []
        self.lvbb_mass = []
        self.true_signal = []
        self.pred_scores = []
        self.targets = []

    def compute_and_log(self, epoch, prefix="val", step=None, log_level=0):
        # print("Accuracy calculated: ", self.accuracy.compute())
        metrics = {
            f"{prefix}/accuracy": self.accuracy.compute(),
            f"{prefix}/auc": self.auc.compute(),
        }
        
        # Add epoch info for proper grouping in W&B
        metrics["epoch"] = epoch
        
        if log_level > 0:
            # Log the background acceptance at different signal selection efficiencies
            signal_metrics = self.signal_selection.compute()
            for level, values in signal_metrics.items():
                metrics.update({
                    f"{prefix}/{level}/lvbb_threshold": values['lvbb_threshold'],
                    f"{prefix}/{level}/qqbb_threshold": values['qqbb_threshold'],
                    f"{prefix}/{level}/bkg_lvbb_fraction": values['bkg_lvbb_fraction'],
                    f"{prefix}/{level}/bkg_qqbb_fraction": values['bkg_qqbb_fraction'],
                })
        if log_level > 1:
            # Confusion matrix plot
            conf_fig = self.confusion.plot(class_names=['Bkg', 'Lep', 'Had'])
            metrics[f"{prefix}/confusion_matrix"] = wandb.Image(conf_fig)
            plt.close(conf_fig)
            # conf_mat = self.confusion.compute().cpu().numpy()
            # fig = plt.figure()
            # plt.imshow(conf_mat, cmap='Blues')
            # plt.colorbar()
            # plt.xlabel("Predicted")
            # plt.ylabel("True")
            # metrics[f"{prefix}/confusion_matrix"] = wandb.Image(fig)
            # plt.close(fig)
        
            # Mass histograms
            if len(self.qqbb_mass) > 0:
                metrics[f"{prefix}/qqbb_mass"] = wandb.Histogram(
                    np.array(self.qqbb_mass), 
                    num_bins=self.mass_bins
                )
            if len(self.lvbb_mass) > 0:
                metrics[f"{prefix}/lvbb_mass"] = wandb.Histogram(
                    np.array(self.lvbb_mass),
                    num_bins=self.mass_bins
                )
        
            # Significance calculation
            if prefix == "val":
                fpr, tpr, thresholds = roc_curve(self.true_signal, self.pred_scores)
                s = tpr * np.sum(np.array(self.true_signal))
                b = fpr * np.sum(~np.array(self.true_signal))
                significance = s / np.sqrt(b + 1e-6)
                best_idx = np.nanargmax(significance)
                
                metrics.update({
                    f"{prefix}/best_significance": significance[best_idx],
                    f"{prefix}/best_threshold": thresholds[best_idx]
                })
                
                # ROC curve
                fig = plt.figure()
                plt.plot(fpr, tpr)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                metrics[f"{prefix}/roc_curve"] = wandb.Image(fig)
                plt.close(fig)

                # wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})

        if config['wandb']:
            wandb.log({**metrics, "epoch": epoch, "step":step})

# Modified loss class with wandb logging
class HEPLoss(nn.Module):
    def __init__(self, alpha=0.1, target_mass=125):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.target_mass = target_mass
        
    def forward(self, inputs, targets, weights, log, masses_qq, masses_lv):
        ce_loss = self.ce(inputs, targets) * weights
        
        # Mass regularization
        # qq_mass_loss = (masses_qq[targets==2] - self.target_mass).pow(2).mean()
        # lv_mass_loss = (masses_lv[targets==1] - self.target_mass).pow(2).mean()
        
        # total_loss = ce_loss.mean() + self.alpha * (qq_mass_loss + lv_mass_loss)
        total_loss = (ce_loss.sum())/(weights.sum())
        
        # Log individual loss components
        if log:
            wandb.log({
                "loss/total": total_loss.item(),
                "loss/ce": ce_loss.mean().item(),
                # "loss/qq_mass": qq_mass_loss.item(),
                # "loss/lv_mass": lv_mass_loss.item()
            }, commit=False)
        
        return total_loss

# %%
model, train_loader, val_loader = models[model_n]['model'], train_dataloader, val_dataloader
log_interval = 5
config = {
        "learning_rate": 1e-4,
        "architecture": "PhysicsTransformer",
        "dataset": "ATLAS_ChargedHiggs",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "wandb":True
    }
if config['wandb']:
    init_wandb(config)
    # wandb.watch(model, log_freq=100)
criterion = HEPLoss()
train_metrics = HEPMetrics()
val_metrics = HEPMetrics()
global_step = 0
for epoch in range(num_epochs):
    # Training phase
    train_loader._reset_indices()
    val_loader._reset_indices()
    model.train()
    n_step = 0
    orig_len_train_dataloader=len(train_loader)
    train_loss_epoch = 0
    sum_weights_epoch = 0
    for batch_idx in range(orig_len_train_dataloader):
        n_step+=1
        if (batch_idx >= orig_len_train_dataloader-5):
            continue
        batch = next(train_loader)
        x, y, w, types, _, mqq, mlv, _ = batch.values()
        x, y, w, types, mqq, mlv = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device)
        
        optimizer.zero_grad()
        outputs = model(x, types)
        loss = criterion(outputs, y, w, config["wandb"], mqq, mlv)
        train_loss_epoch += loss.item()
        sum_weights_epoch += w.sum().item()
        
        loss.backward()
        optimizer.step()
        
        # Update training metrics
        train_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv)
        if (n_step % 10) == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, orig_len_train_dataloader, loss.item()))
            # Log training metrics every log_interval batches
        if ((batch_idx % log_interval) == 0):
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=1)
            train_metrics.reset()  # Reset after logging to track fresh metrics
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if config['wandb']:
                wandb.log({"train/lr": current_lr}, step=global_step)
            
            # # Log sample predictions
            # sample_probs = F.softmax(outputs[:5], dim=1).detach().cpu().numpy()
            # sample_preds = outputs[:5].argmax(dim=1).detach().cpu().numpy()
            # sample_targets = y[:5].detach().cpu().numpy()
            # if config['wandb']:
            #     wandb.log({
            #         "train/sample_predictions": wandb.Table(
            #             columns=["Target", "Predicted", "Probabilities"],
            #             data=[
            #                 [sample_targets[i], sample_preds[i], sample_probs[i]] 
            #                 for i in range(len(sample_targets))
            #             ]
            #         )
            #     }, step=global_step)
        
        global_step += 1
    if config['wandb']:
        wandb.log({'train/loss_total':train_loss_epoch/sum_weights_epoch}, commit=False)
        train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=2)
        train_metrics.reset()  # Reset after logging to track fresh metrics
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"train/lr": current_lr}, step=global_step)
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

            x, y, w, types, _, mqq, mlv, _ = batch.values()
            x, y, w, types, mqq, mlv = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device)
            
            outputs = model(x, types)
            loss += criterion(outputs, y, w, config["wandb"], mqq, mlv).sum()
            wt_sum += w.sum()
            val_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv)
            # print('[%d/%d][%d/%d] Val' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader))
        if config['wandb']:
            wandb.log({
                "val/loss_total": loss.item()/wt_sum.item(),
                "val/loss_ce": loss.item()/wt_sum.item(),
                # "loss/qq_mass": qq_mass_loss.item(),
                # "loss/lv_mass": lv_mass_loss.item()
            })
        print('[%d/%d][%d/%d]\tVAL Loss_C: %.4e' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader, loss.item()/wt_sum.item()))
    
    # Log validation metrics
    if config['wandb']:
        val_metrics.compute_and_log(epoch, prefix="val", step=global_step, log_level=2)
        val_metrics.reset()

    
    # Log model gradients and parameters
    if 1:
        if epoch % 5 == 0:
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

# %%

class SignalSelectionMetrics:
    def __init__(self, signal_acceptance_levels=[0.5, 0.7, 0.9]):
        """
        Args:
            signal_acceptance_levels: List of percentages (0-1) of signal events to accept
        """
        self.signal_levels = sorted(signal_acceptance_levels)
        self.reset()
        
    def reset(self):
        # Store all predictions and targets with weights
        self.all_probs = []
        self.all_targets = []
        self.all_weights = []
        
    def update(self, preds, targets, weights):
        """
        Args:
            preds: Tensor of shape [batch_size, 3] with predicted probabilities
                  (columns: background, lvbb, qqbb)
            targets: Tensor of shape [batch_size] with true class indices
            weights: Tensor of shape [batch_size] with sample weights
        """
        # Convert to probabilities if needed
        if preds.shape[1] > 3:  # If logits are passed
            probs = F.softmax(preds, dim=1)
        else:
            probs = preds
            
        self.all_probs.append(probs.cpu())
        self.all_targets.append(targets.cpu())
        self.all_weights.append(weights.cpu())
        
    def compute(self):
        if not self.all_probs:
            return {}
            
        # Concatenate all batches
        probs = torch.cat(self.all_probs)
        targets = torch.cat(self.all_targets)
        weights = torch.cat(self.all_weights)
        
        # Separate signal and background
        bkg_mask = targets == 0
        lvbb_mask = targets == 1
        qqbb_mask = targets == 2
        
        # Get probabilities
        p_bkg = probs[:, 0]
        p_lvbb = probs[:, 1]
        p_qqbb = probs[:, 2]
        
        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for each signal acceptance level
        for level in self.signal_levels:
            # Calculate thresholds for this acceptance level
            lvbb_thresh = self._find_threshold(
                p_bkg[lvbb_mask], p_lvbb[lvbb_mask], p_qqbb[lvbb_mask],
                weights[lvbb_mask], level
            )
            qqbb_thresh = self._find_threshold(
                p_bkg[qqbb_mask], p_qqbb[qqbb_mask], p_lvbb[qqbb_mask],
                weights[qqbb_mask], level
            )
            
            # Apply selection logic
            lvbb_selected = (
                (p_lvbb >= p_qqbb) & 
                (p_bkg < lvbb_thresh)
            )
            qqbb_selected = (
                (p_qqbb >= p_lvbb) & 
                (p_bkg < qqbb_thresh)
            )
            
            # Calculate background proportions
            bkg_lvbb = (bkg_mask & lvbb_selected).float() * weights
            bkg_qqbb = (bkg_mask & qqbb_selected).float() * weights
            
            total_bkg_weight = weights[bkg_mask].sum()
            
            if total_bkg_weight > 0:
                bkg_lvbb_frac = bkg_lvbb.sum() / total_bkg_weight
                bkg_qqbb_frac = bkg_qqbb.sum() / total_bkg_weight
            else:
                bkg_lvbb_frac = 0.0
                bkg_qqbb_frac = 0.0
                
            # Store results
            results[f'signal_acceptance_{int(level*100)}'] = {
                'lvbb_threshold': lvbb_thresh,
                'qqbb_threshold': qqbb_thresh,
                'bkg_lvbb_fraction': bkg_lvbb_frac.item(),
                'bkg_qqbb_fraction': bkg_qqbb_frac.item()
            }
            
        return results

    def _find_threshold(self, p_bkg, p_sig, p_other_sig, weights, acceptance_level):#, signal_type):
        """
        Find the background probability threshold that accepts the specified
        fraction of signal events, while respecting the mutual exclusivity rule.
        
        Args:
            p_bkg: Background probabilities for signal events
            p_sig: Signal probabilities (lvbb or qqbb) for signal events
            p_other_sig: Other signal probabilities (qqbb or lvbb) for signal events
            weights: Event weights
            acceptance_level: Desired fraction of signal to accept
            signal_type: 'lvbb' or 'qqbb' - determines which exclusivity rule to apply
        """
        if len(p_sig) == 0:
            return 1.0  # No signal events
        
        # Apply mutual exclusivity rule
        valid_mask = p_sig >= p_other_sig
        # if signal_type == 'lvbb':
        #     valid_mask = p_sig >= p_other_sig
        # else:  # qqbb
        #     valid_mask = p_sig >= p_other_sig
            
        # Filter events that satisfy the mutual exclusivity rule
        p_bkg = p_bkg[valid_mask]
        p_sig = p_sig[valid_mask]
        weights = weights[valid_mask]
        
        if len(p_sig) == 0:
            return 1.0  # No signal events after applying exclusivity
        
        # Sort signal events by p_bkg
        sorted_idx = torch.argsort(p_bkg)
        sorted_p_bkg = p_bkg[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Calculate cumulative sum of weights
        cum_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = cum_weights[-1]
        
        # Find threshold that accepts desired fraction
        target_weight = total_weight * acceptance_level
        idx = torch.searchsorted(cum_weights, target_weight)
        
        if idx >= len(sorted_p_bkg):
            return 1.0
        return sorted_p_bkg[idx].item()

class WeightedMulticlassAccuracy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.total_correct = 0.0
        self.total_weight = 0.0
        
    def update(self, preds, targets, weights=None):
        """
        Args:
            preds: Tensor of shape [batch_size, num_classes] with predicted logits
            targets: Tensor of shape [batch_size] with true class indices
            weights: Optional tensor of shape [batch_size] with sample weights
        """
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
            
        # Convert predictions to class indices
        pred_classes = torch.argmax(preds, dim=1)
        
        # Calculate weighted correct predictions
        correct = (pred_classes == targets).float()
        weighted_correct = correct * weights
        
        self.total_correct += weighted_correct.sum().item()
        self.total_weight += weights.sum().item()
        
    def compute(self):
        if self.total_weight == 0:
            return 0.0
        return self.total_correct / self.total_weight

class WeightedMulticlassAUC:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.all_probs = []
        self.all_targets = []
        self.all_weights = []
        
    def update(self, preds, targets, weights=None):
        """
        Args:
            preds: Tensor of shape [batch_size, num_classes] with predicted logits
            targets: Tensor of shape [batch_size] with true class indices
            weights: Optional tensor of shape [batch_size] with sample weights
        """
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
            
        probs = F.softmax(preds, dim=1)
        
        # Store for later computation
        self.all_probs.append(probs.cpu())
        self.all_targets.append(targets.cpu())
        self.all_weights.append(weights.cpu())
        
    def compute(self):
        if not self.all_probs:
            return 0.0
            
        # Concatenate all batches
        probs = torch.cat(self.all_probs)
        targets = torch.cat(self.all_targets)
        weights = torch.cat(self.all_weights)
        
        # Compute one-vs-rest AUC for each class
        auc_scores = []
        for class_idx in range(self.num_classes):
            # Create binary targets for this class
            binary_targets = (targets == class_idx).float()
            
            # Get probabilities for this class
            class_probs = probs[:, class_idx]
            
            # Compute weighted AUC
            if len(torch.unique(binary_targets)) < 2:
                # Only one class present, skip
                continue
                
            # Sort by predicted probability
            sort_idx = torch.argsort(class_probs)
            sorted_probs = class_probs[sort_idx]
            sorted_targets = binary_targets[sort_idx]
            sorted_weights = weights[sort_idx]
            
            # Compute weighted TPR and FPR
            total_pos_weight = (sorted_targets * sorted_weights).sum()
            total_neg_weight = ((1 - sorted_targets) * sorted_weights).sum()
            
            tpr = torch.cumsum(sorted_targets * sorted_weights, dim=0) / total_pos_weight
            fpr = torch.cumsum((1 - sorted_targets) * sorted_weights, dim=0) / total_neg_weight
            
            # Compute AUC using trapezoidal rule
            auc = torch.trapz(tpr, fpr).item()
            auc_scores.append(auc)
        
        if not auc_scores:
            return 0.0
        return sum(auc_scores) / len(auc_scores)

class WeightedConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.float32
        ).cpu()
        
    def update(self, preds, targets, weights=None):
        """
        Args:
            preds: Tensor of shape [batch_size, num_classes] with predicted logits
            targets: Tensor of shape [batch_size] with true class indices
            weights: Optional tensor of shape [batch_size] with sample weights
        """
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
            
        # Convert predictions to class indices
        pred_classes = torch.argmax(preds, dim=1)
        
        # Create one-hot encoding of predictions and targets
        pred_onehot = F.one_hot(pred_classes, num_classes=self.num_classes).float()
        target_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Outer product to get confusion matrix contributions
        batch_conf = torch.matmul(
            target_onehot.transpose(0, 1),  # [num_classes, batch_size]
            pred_onehot * weights.unsqueeze(-1)  # [batch_size, num_classes]
        )
        
        self.conf_matrix += batch_conf.cpu()
        
    def compute(self):
        return self.conf_matrix
    
    def plot(self, class_names=None):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
            class_names: List of class names for labeling axes
        """
        conf_matrix = self.conf_matrix.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(conf_matrix, cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Add labels
        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
            
        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Confusion Matrix",
            ylabel="True label",
            xlabel="Predicted label"
        )
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if conf_matrix.dtype.kind == 'f' else 'd'
        thresh = conf_matrix.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, format(conf_matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig

def init_wandb(config):
    wandb.init(
        project="HEP-Transformers",
        config=config,
        magic=True
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")


class HEPMetrics:
    def __init__(self, 
                 num_classes=3, 
                 mass_bins=50, 
                 mass_range=(0, 1000), 
                 signal_acceptance_levels=[0.5, 0.7, 0.9]
                 ):
        # Classification metrics
        self.accuracy = WeightedMulticlassAccuracy(num_classes=num_classes)
        self.auc = WeightedMulticlassAUC(num_classes=num_classes)
        self.confusion = WeightedConfusionMatrix(num_classes=num_classes)
        self.confusion = WeightedConfusionMatrix(num_classes=num_classes)
        self.signal_selection = SignalSelectionMetrics(signal_acceptance_levels)
        
        
        # Mass reconstruction histograms
        self.mass_bins = mass_bins
        self.mass_range = mass_range
        self.qqbb_mass = []
        self.lvbb_mass = []
        
        # Significance tracking
        self.true_signal = []
        self.pred_scores = []
        self.targets = []
    
    def update(self, preds, targets, weights, masses_qq, masses_lv):
        probs = F.softmax(preds, dim=1)
        self.accuracy.update(preds, targets, weights)
        # print("Accuracy calculated here: ", self.accuracy.compute())
        self.auc.update(preds, targets, weights)
        self.confusion.update(preds, targets, weights)
        self.signal_selection.update(preds, targets, weights)
        
        # Store masses for correctly classified signals
        sig_mask = targets > 0
        pred_labels = preds.argmax(dim=1)
        
        qq_correct = (targets == 2) & (pred_labels == 2)
        lv_correct = (targets == 1) & (pred_labels == 1)
        
        self.qqbb_mass.extend(masses_qq[qq_correct].cpu().tolist())
        self.lvbb_mass.extend(masses_lv[lv_correct].cpu().tolist())
        
        # For significance calculation
        self.true_signal.extend((targets > 0).cpu().tolist())
        self.pred_scores.extend(probs[:,1:].sum(dim=1).cpu().tolist())
        self.targets.extend(targets.cpu().tolist())
    
    def reset(self):
        # Reset metrics
        self.accuracy.reset()
        self.auc.reset()
        self.confusion.reset()
        self.qqbb_mass = []
        self.lvbb_mass = []
        self.true_signal = []
        self.pred_scores = []
        self.targets = []

    def compute_and_log(self, epoch, prefix="val", step=None, log_level=0):
        # print("Accuracy calculated: ", self.accuracy.compute())
        metrics = {
            f"{prefix}/accuracy": self.accuracy.compute(),
            f"{prefix}/auc": self.auc.compute(),
        }
        
        # Add epoch info for proper grouping in W&B
        metrics["epoch"] = epoch
        
        if log_level > 0:
            # Log the background acceptance at different signal selection efficiencies
            signal_metrics = self.signal_selection.compute()
            for level, values in signal_metrics.items():
                metrics.update({
                    f"{prefix}/{level}/lvbb_threshold": values['lvbb_threshold'],
                    f"{prefix}/{level}/qqbb_threshold": values['qqbb_threshold'],
                    f"{prefix}/{level}/bkg_lvbb_fraction": values['bkg_lvbb_fraction'],
                    f"{prefix}/{level}/bkg_qqbb_fraction": values['bkg_qqbb_fraction'],
                })
        if log_level > 1:
            # Confusion matrix plot
            conf_fig = self.confusion.plot(class_names=['Bkg', 'Lep', 'Had'])
            metrics[f"{prefix}/confusion_matrix"] = wandb.Image(conf_fig)
            plt.close(conf_fig)
            # conf_mat = self.confusion.compute().cpu().numpy()
            # fig = plt.figure()
            # plt.imshow(conf_mat, cmap='Blues')
            # plt.colorbar()
            # plt.xlabel("Predicted")
            # plt.ylabel("True")
            # metrics[f"{prefix}/confusion_matrix"] = wandb.Image(fig)
            # plt.close(fig)
        
            # Mass histograms
            if len(self.qqbb_mass) > 0:
                metrics[f"{prefix}/qqbb_mass"] = wandb.Histogram(
                    np.array(self.qqbb_mass), 
                    num_bins=self.mass_bins
                )
            if len(self.lvbb_mass) > 0:
                metrics[f"{prefix}/lvbb_mass"] = wandb.Histogram(
                    np.array(self.lvbb_mass),
                    num_bins=self.mass_bins
                )
        
            # Significance calculation
            if prefix == "val":
                fpr, tpr, thresholds = roc_curve(self.true_signal, self.pred_scores)
                s = tpr * np.sum(np.array(self.true_signal))
                b = fpr * np.sum(~np.array(self.true_signal))
                significance = s / np.sqrt(b + 1e-6)
                best_idx = np.nanargmax(significance)
                
                metrics.update({
                    f"{prefix}/best_significance": significance[best_idx],
                    f"{prefix}/best_threshold": thresholds[best_idx]
                })
                
                # ROC curve
                fig = plt.figure()
                plt.plot(fpr, tpr)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                metrics[f"{prefix}/roc_curve"] = wandb.Image(fig)
                plt.close(fig)

                # wandb.log({"pr": wandb.plot.pr_curve(ground_truth, predictions)})

        if config['wandb']:
            wandb.log({**metrics, "epoch": epoch, "step":step})

# Modified loss class with wandb logging
class HEPLoss(nn.Module):
    def __init__(self, alpha=0.1, target_mass=125):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.target_mass = target_mass
        
    def forward(self, inputs, targets, weights, log, masses_qq, masses_lv):
        ce_loss = self.ce(inputs, targets) * weights
        
        # Mass regularization
        # qq_mass_loss = (masses_qq[targets==2] - self.target_mass).pow(2).mean()
        # lv_mass_loss = (masses_lv[targets==1] - self.target_mass).pow(2).mean()
        
        # total_loss = ce_loss.mean() + self.alpha * (qq_mass_loss + lv_mass_loss)
        total_loss = (ce_loss.sum())/(weights.sum())
        
        # Log individual loss components
        if log:
            wandb.log({
                "loss/total": total_loss.item(),
                "loss/ce": ce_loss.mean().item(),
                # "loss/qq_mass": qq_mass_loss.item(),
                # "loss/lv_mass": lv_mass_loss.item()
            }, commit=False)
        
        return total_loss

# %%
model, train_loader, val_loader = models[model_n]['model'], train_dataloader, val_dataloader
log_interval = 5
config = {
        "learning_rate": 1e-4,
        "architecture": "PhysicsTransformer",
        "dataset": "ATLAS_ChargedHiggs",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "wandb":True
    }
if config['wandb']:
    init_wandb(config)
    # wandb.watch(model, log_freq=100)
criterion = HEPLoss()
train_metrics = HEPMetrics()
val_metrics = HEPMetrics()
global_step = 0
for epoch in range(num_epochs):
    # Training phase
    train_loader._reset_indices()
    val_loader._reset_indices()
    model.train()
    n_step = 0
    orig_len_train_dataloader=len(train_loader)
    train_loss_epoch = 0
    sum_weights_epoch = 0
    for batch_idx in range(orig_len_train_dataloader):
        n_step+=1
        if (batch_idx >= orig_len_train_dataloader-5):
            continue
        batch = next(train_loader)
        x, y, w, types, _, mqq, mlv, _ = batch.values()
        x, y, w, types, mqq, mlv = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device)
        
        optimizer.zero_grad()
        outputs = model(x, types)
        loss = criterion(outputs, y, w, config["wandb"], mqq, mlv)
        train_loss_epoch += loss.item()
        sum_weights_epoch += w.sum().item()
        
        loss.backward()
        optimizer.step()
        
        # Update training metrics
        train_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv)
        if (n_step % 10) == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4e' %(epoch, num_epochs, n_step, orig_len_train_dataloader, loss.item()))
            # Log training metrics every log_interval batches
        if ((batch_idx % log_interval) == 0):
            train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=1)
            train_metrics.reset()  # Reset after logging to track fresh metrics
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if config['wandb']:
                wandb.log({"train/lr": current_lr}, step=global_step)
            
            # # Log sample predictions
            # sample_probs = F.softmax(outputs[:5], dim=1).detach().cpu().numpy()
            # sample_preds = outputs[:5].argmax(dim=1).detach().cpu().numpy()
            # sample_targets = y[:5].detach().cpu().numpy()
            # if config['wandb']:
            #     wandb.log({
            #         "train/sample_predictions": wandb.Table(
            #             columns=["Target", "Predicted", "Probabilities"],
            #             data=[
            #                 [sample_targets[i], sample_preds[i], sample_probs[i]] 
            #                 for i in range(len(sample_targets))
            #             ]
            #         )
            #     }, step=global_step)
        
        global_step += 1
    if config['wandb']:
        wandb.log({'train/loss_total':train_loss_epoch/sum_weights_epoch}, commit=False)
        train_metrics.compute_and_log(epoch, prefix="train", step=global_step, log_level=2)
        train_metrics.reset()  # Reset after logging to track fresh metrics
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"train/lr": current_lr}, step=global_step)
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

            x, y, w, types, _, mqq, mlv, _ = batch.values()
            x, y, w, types, mqq, mlv = x.to(device), y.to(device), w.to(device), types.to(device), mqq.to(device), mlv.to(device)
            
            outputs = model(x, types)
            loss += criterion(outputs, y, w, config["wandb"], mqq, mlv).sum()
            wt_sum += w.sum()
            val_metrics.update(outputs, y.argmax(dim=-1), w, mqq, mlv)
            # print('[%d/%d][%d/%d] Val' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader))
        if config['wandb']:
            wandb.log({
                "val/loss_total": loss.item()/wt_sum.item(),
                "val/loss_ce": loss.item()/wt_sum.item(),
                # "loss/qq_mass": qq_mass_loss.item(),
                # "loss/lv_mass": lv_mass_loss.item()
            })
        print('[%d/%d][%d/%d]\tVAL Loss_C: %.4e' %(epoch, num_epochs, batch_idx, orig_len_val_dataloader, loss.item()/wt_sum.item()))
    
    # Log validation metrics
    if config['wandb']:
        val_metrics.compute_and_log(epoch, prefix="val", step=global_step, log_level=2)
        val_metrics.reset()

    
    # Log model gradients and parameters
    if 1:
        if epoch % 5 == 0:
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


