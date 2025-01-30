import torch 
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

class ByMassSignalSelectionMetrics:
    def __init__(self, total_weights_per_dsid={}, signal_acceptance_levels=[0.5, 0.7, 0.9]):
        """
        Args:
            signal_acceptance_levels: List of percentages (0-1) of signal events to accept
        """
        assert(len(total_weights_per_dsid))
        self.total_weights_per_dsid = total_weights_per_dsid
        self.signal_levels = sorted(signal_acceptance_levels)
        self.reset()
        
    def reset(self):
        # Store all predictions and targets with weights
        self.all_probs = []
        self.all_targets = []
        self.all_weights = []
        self.all_dsid = []  # to track dsid for each sample
        self.dsid_weight_sums = {}  # to track weight sums per dsid
        
    def update(self, preds, targets, weights, dsid):
        """
        Args:
            preds: Tensor of shape [batch_size, 3] with predicted probabilities
                  (columns: background, lvbb, qqbb)
            targets: Tensor of shape [batch_size] with true class indices
            weights: Tensor of shape [batch_size] with sample weights
            dsid: Tensor of shape [batch_size] with dsid for each sample
        """
        # Convert to probabilities if needed
        if (preds.sum(dim=-1).mean() != 1):  # If logits are passed
            probs = F.softmax(preds, dim=1)
        else:
            probs = preds
            
        self.all_probs.append(probs.cpu())
        self.all_targets.append(targets.cpu())
        self.all_weights.append(weights.cpu())
        self.all_dsid.append(dsid.cpu())

        # Update weight sums per dsid
        unique_dsid = dsid.cpu().unique()
        for d in unique_dsid:
            if d not in self.dsid_weight_sums:
                self.dsid_weight_sums[d.item()] = 0.0
            self.dsid_weight_sums[d.item()] += weights[dsid == d].sum().item()
        
    def compute(self):
        if not self.all_probs:
            return {}
            
        # Concatenate all batches
        probs = torch.cat(self.all_probs)
        targets = torch.cat(self.all_targets)
        weights = torch.cat(self.all_weights)
        dsids = torch.cat(self.all_dsid)
        
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
        DSID_MASS_MAPPING = {510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
        MASS_DSID_MAPPING = {v: k for k, v in DSID_MASS_MAPPING.items()} # Create inverse dictionary
        for level in self.signal_levels:
            for signal_dsid in DSID_MASS_MAPPING.keys():
                signal_mass_mask = dsids == signal_dsid
                # Calculate thresholds for this acceptance level
                lvbb_thresh = self._find_threshold(
                    p_bkg[lvbb_mask & signal_mass_mask], p_lvbb[lvbb_mask & signal_mass_mask], p_qqbb[lvbb_mask & signal_mass_mask],
                    weights[lvbb_mask & signal_mass_mask], level
                )
                qqbb_thresh = self._find_threshold(
                    p_bkg[qqbb_mask & signal_mass_mask], p_qqbb[qqbb_mask & signal_mass_mask], p_lvbb[qqbb_mask & signal_mass_mask],
                    weights[qqbb_mask & signal_mass_mask], level
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

                if 0: # Old
                    total_bkg_weight = weights[bkg_mask].sum()
                    if total_bkg_weight > 0:
                        bkg_lvbb_frac = bkg_lvbb.sum() / total_bkg_weight
                        bkg_qqbb_frac = bkg_qqbb.sum() / total_bkg_weight
                    else:
                        bkg_lvbb_frac = 0.0
                        bkg_qqbb_frac = 0.0
                else: # New, do total not just probability
                    # Initialize a dictionary to store results for each dsid
                    bkg_lvbb_exp_per_dsid = {}
                    bkg_qqbb_exp_per_dsid = {}
                    
                    # Calculate for each dsid
                    for dsid in dsids.unique():
                        # Filter by dsid
                        bkg_lvbb_dsid = bkg_lvbb[dsids == dsid]
                        bkg_qqbb_dsid = bkg_qqbb[dsids == dsid]
                        total_processed_weight_dsid = self.dsid_weight_sums.get(dsid.item(), 0.0)
                        
                        if total_processed_weight_dsid !=0:
                            bkg_lvbb_exp_per_dsid[dsid] = bkg_lvbb_dsid.sum().item() / total_processed_weight_dsid * self.total_weights_per_dsid[dsid.item()]
                            bkg_qqbb_exp_per_dsid[dsid] = bkg_qqbb_dsid.sum().item() / total_processed_weight_dsid  * self.total_weights_per_dsid[dsid.item()]
                        else:
                            bkg_lvbb_exp_per_dsid[dsid] = 0.0
                            bkg_qqbb_exp_per_dsid[dsid] = 0.0
                    
                # Store results
                results[(level, signal_dsid)] = {
                    'lvbb_threshold': lvbb_thresh,
                    'qqbb_threshold': qqbb_thresh,
                    'bkg_lvbb_expected': sum(bkg_lvbb_exp_per_dsid.values()),# if lvbb_thresh != 1 else np.nan, # Guard against the case where the threshold is set to accept all
                    'bkg_qqbb_expected': sum(bkg_qqbb_exp_per_dsid.values())# if qqbb_thresh != 1 else np.nan
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
        sum_wts_failing_mutual_exclusivity = weights[~valid_mask].sum()
        p_bkg = p_bkg[valid_mask]
        p_sig = p_sig[valid_mask]
        weights = weights[valid_mask]
        
        if len(p_sig) == 0:
            return 1.0  # No signal events after applying exclusivity
        
        # Sort remaining signal events by p_bkg
        sorted_idx = torch.argsort(p_bkg)
        sorted_p_bkg = p_bkg[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Calculate cumulative sum of weights
        cum_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = cum_weights[-1]
        
        # Find threshold that accepts desired fraction
        target_weight = (total_weight + sum_wts_failing_mutual_exclusivity) * acceptance_level
        idx = torch.searchsorted(cum_weights, target_weight)
        
        if idx >= len(sorted_p_bkg):
            return 1.0
            # return np.nan
        return sorted_p_bkg[idx].item()

class SignalSelectionMetrics:
    def __init__(self, total_weights_per_dsid={}, signal_acceptance_levels=[0.5, 0.7, 0.9]):
        """
        Args:
            signal_acceptance_levels: List of percentages (0-1) of signal events to accept
        """
        assert(len(total_weights_per_dsid))
        self.total_weights_per_dsid = total_weights_per_dsid
        self.signal_levels = sorted(signal_acceptance_levels)
        self.reset()
        
    def reset(self):
        # Store all predictions and targets with weights
        self.all_probs = []
        self.all_targets = []
        self.all_weights = []
        self.all_dsid = []  # to track dsid for each sample
        self.dsid_weight_sums = {}  # to track weight sums per dsid
        
    def update(self, preds, targets, weights, dsid):
        """
        Args:
            preds: Tensor of shape [batch_size, 3] with predicted probabilities
                  (columns: background, lvbb, qqbb)
            targets: Tensor of shape [batch_size] with true class indices
            weights: Tensor of shape [batch_size] with sample weights
            dsid: Tensor of shape [batch_size] with dsid for each sample
        """
        # Convert to probabilities if needed
        if (preds.sum(dim=-1).mean() != 1):  # If logits are passed
            probs = F.softmax(preds, dim=1)
        else:
            probs = preds
            
        self.all_probs.append(probs.cpu())
        self.all_targets.append(targets.cpu())
        self.all_weights.append(weights.cpu())
        self.all_dsid.append(dsid.cpu())

        # Update weight sums per dsid
        unique_dsid = dsid.cpu().unique()
        for d in unique_dsid:
            if d not in self.dsid_weight_sums:
                self.dsid_weight_sums[d.item()] = 0.0
            self.dsid_weight_sums[d.item()] += weights[dsid == d].sum().item()
        
    def compute(self):
        if not self.all_probs:
            return {}
            
        # Concatenate all batches
        probs = torch.cat(self.all_probs)
        targets = torch.cat(self.all_targets)
        weights = torch.cat(self.all_weights)
        dsids = torch.cat(self.all_dsid)
        
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

            if 0: # Old
                total_bkg_weight = weights[bkg_mask].sum()
                if total_bkg_weight > 0:
                    bkg_lvbb_frac = bkg_lvbb.sum() / total_bkg_weight
                    bkg_qqbb_frac = bkg_qqbb.sum() / total_bkg_weight
                else:
                    bkg_lvbb_frac = 0.0
                    bkg_qqbb_frac = 0.0
            else: # New, do total not just probability
                # Initialize a dictionary to store results for each dsid
                bkg_lvbb_exp_per_dsid = {}
                bkg_qqbb_exp_per_dsid = {}
                
                # Calculate for each dsid
                for dsid in dsids.unique():
                    # Filter by dsid
                    bkg_lvbb_dsid = bkg_lvbb[dsids == dsid]
                    bkg_qqbb_dsid = bkg_qqbb[dsids == dsid]
                    total_processed_weight_dsid = self.dsid_weight_sums.get(dsid.item(), 0.0)
                    
                    if total_processed_weight_dsid !=0:
                        bkg_lvbb_exp_per_dsid[dsid] = bkg_lvbb_dsid.sum().item() / total_processed_weight_dsid * self.total_weights_per_dsid[dsid.item()]
                        bkg_qqbb_exp_per_dsid[dsid] = bkg_qqbb_dsid.sum().item() / total_processed_weight_dsid  * self.total_weights_per_dsid[dsid.item()]
                    else:
                        bkg_lvbb_exp_per_dsid[dsid] = 0.0
                        bkg_qqbb_exp_per_dsid[dsid] = 0.0
                
            # Store results
            results[f'signal_acceptance_{int(level*100)}'] = {
                'lvbb_threshold': lvbb_thresh,
                'qqbb_threshold': qqbb_thresh,
                'bkg_lvbb_expected': sum(bkg_lvbb_exp_per_dsid.values()),# if lvbb_thresh != 1 else np.nan, # Guard against the case where the threshold is set to accept all
                'bkg_qqbb_expected': sum(bkg_qqbb_exp_per_dsid.values())# if qqbb_thresh != 1 else np.nan
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
        sum_wts_failing_mutual_exclusivity = weights[~valid_mask].sum()
        p_bkg = p_bkg[valid_mask]
        p_sig = p_sig[valid_mask]
        weights = weights[valid_mask]
        
        if len(p_sig) == 0:
            return 1.0  # No signal events after applying exclusivity
        
        # Sort remaining signal events by p_bkg
        sorted_idx = torch.argsort(p_bkg)
        sorted_p_bkg = p_bkg[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Calculate cumulative sum of weights
        cum_weights = torch.cumsum(sorted_weights, dim=0)
        total_weight = cum_weights[-1]
        
        # Find threshold that accepts desired fraction
        target_weight = (total_weight + sum_wts_failing_mutual_exclusivity) * acceptance_level
        idx = torch.searchsorted(cum_weights, target_weight)
        
        if idx >= len(sorted_p_bkg):
            return 1.0
            # return np.nan
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
        name=config["name"],
        magic=True,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")


class HEPMetrics:
    def __init__(self, 
                 num_classes=3, 
                 mass_bins=50, 
                 mass_range=(0, 1000), 
                 signal_acceptance_levels=[0.5, 0.7, 0.9],
                 total_weights_per_dsid={},
                 ):
        # Classification metrics
        self.accuracy = WeightedMulticlassAccuracy(num_classes=num_classes)
        self.auc = WeightedMulticlassAUC(num_classes=num_classes)
        self.confusion = WeightedConfusionMatrix(num_classes=num_classes)
        self.signal_selection = SignalSelectionMetrics(total_weights_per_dsid, signal_acceptance_levels)
        self.by_mass_signal_selection = ByMassSignalSelectionMetrics(total_weights_per_dsid, signal_acceptance_levels)
        
        
        # Mass reconstruction histograms
        self.mass_bins = mass_bins
        self.mass_range = mass_range
        self.qqbb_mass = []
        self.lvbb_mass = []
        
        # Significance tracking
        self.true_signal = []
        self.pred_scores = []
        # self.targets = []
    
    def update(self, preds, targets, weights, masses_qq, masses_lv, dsid):
        probs = F.softmax(preds, dim=1)
        self.accuracy.update(preds, targets, weights)
        # print("Accuracy calculated here: ", self.accuracy.compute())
        self.auc.update(preds, targets, weights)
        self.confusion.update(preds, targets, weights)
        self.signal_selection.update(preds, targets, weights, dsid)
        self.by_mass_signal_selection.update(preds, targets, weights, dsid)
        
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
        # self.targets.extend(targets.cpu().tolist())
    
    def reset(self, log_level):
        # Reset metrics
        self.accuracy.reset()
        self.auc.reset()
        if log_level > 0:
            self.signal_selection.reset()
        if log_level > 1:
            self.confusion.reset()
            self.by_mass_signal_selection.reset()
            self.qqbb_mass = []
            self.lvbb_mass = []
            self.true_signal = []
            self.pred_scores = []
        # self.targets = []

    def compute_and_log(self, epoch, prefix="val", step=None, log_level=0, save=True):
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
                    f"{prefix}/{level}/bkg_lvbb_expected": values['bkg_lvbb_expected'],
                    f"{prefix}/{level}/bkg_qqbb_expected": values['bkg_qqbb_expected'],
                })
        if log_level > 1:
            DSID_MASS_MAPPING = {510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
            MASS_DSID_MAPPING = {v: k for k, v in DSID_MASS_MAPPING.items()} # Create inverse dictionary
            signal_metrics = self.by_mass_signal_selection.compute()
            for level_dsid, values in signal_metrics.items():
                metrics.update({
                    f"{prefix}_ByMassAcceptance/lvbb_threshold/{level_dsid[0]}_{DSID_MASS_MAPPING[level_dsid[1]]}": values['lvbb_threshold'],
                    f"{prefix}_ByMassAcceptance/qqbb_threshold/{level_dsid[0]}_{DSID_MASS_MAPPING[level_dsid[1]]}": values['qqbb_threshold'],
                    f"{prefix}_ByMassAcceptance/bkg_lvbb_expected/{level_dsid[0]}_{DSID_MASS_MAPPING[level_dsid[1]]}": values['bkg_lvbb_expected'],
                    f"{prefix}_ByMassAcceptance/bkg_qqbb_expected/{level_dsid[0]}_{DSID_MASS_MAPPING[level_dsid[1]]}": values['bkg_qqbb_expected'],
                })
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

        if save:
            wandb.log({**metrics, "epoch": epoch, "step":step})
        else:
            print(metrics)

# Modified loss class with wandb logging
class HEPLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, target_mass=125):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
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
                # "loss/ce": ce_loss.mean().item(),
                # "loss/qq_mass": qq_mass_loss.item(),
                # "loss/lv_mass": lv_mass_loss.item()
            }, commit=False)
        
        return total_loss