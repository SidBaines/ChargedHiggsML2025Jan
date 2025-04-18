import torch 
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from utils import weighted_correlation

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
                 channel=None, 
                 max_buffer_len=100000,
                 mass_bins=50, 
                 mass_range=(0, 1000), 
                 signal_acceptance_levels=[500],
                 max_bkg_levels=[200],
                 total_weights_per_dsid={},
                 mHLimits=[(0,1e10),(95e3, 140e3)],
                 unweighted=False,
                 ):
        assert((channel=="lvbb") or (channel=="qqbb"))
        self.channel=channel
        self.num_classes = 2
        self.max_buffer_len = max_buffer_len
        self.signal_acceptance_levels = sorted(signal_acceptance_levels)
        self.max_bkg_levels = sorted(max_bkg_levels)
        assert(len(total_weights_per_dsid))
        self.total_weights_per_dsid = total_weights_per_dsid
        self.mH_Limits = mHLimits
        self.unweighted = unweighted
        self.class_labels = {0:"bkg", 1:channel}
        self.DSID_MASS_MAPPING = {510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
        
        # Mass reconstruction histograms
        # self.mass_bins = mass_bins
        # self.mass_range = mass_range
        # self.qqbb_mass = []
        # self.lvbb_mass = []
        self.reset()
        
    def reset(self):
        # Store all predictions and targets with weights
        self.all_probs = np.zeros((self.max_buffer_len, self.num_classes))
        self.all_targets = np.zeros(self.max_buffer_len)
        self.all_weights = np.zeros(self.max_buffer_len)
        self.all_dsids = np.zeros(self.max_buffer_len).astype(int)  # to track dsid for each sample
        self.all_mHs = np.zeros(self.max_buffer_len)  # to track dsid for each sample
        self.all_mWhs = np.zeros(self.max_buffer_len)  # to track dsid for each sample
        self.processed_weight_sums_per_dsid = {}  # to track weight sums per dsid
        self.current_update_point = 0
        self.starts = {
            'accuracy' : 0,
            'auc' : 0,
            'conf' : 0,
            'sig_sel' : 0,
        }

    def reset_starts(self, ks=None):
        if ks is None: # All of them
            self.starts = {k:0 for k in self.starts.keys()}
        else:
            for k in ks:
                self.starts[k] = 0

    def update(self, preds, targets, weights, mWh, dsid, mH):
        """
        Args:
            preds: Tensor of shape [batch_size, 3] with predicted probabilities
                  (columns: background, lvbb, qqbb)
            targets: Tensor of shape [batch_size] with true class indices
            weights: Tensor of shape [batch_size] with sample weights
            dsid: Tensor of shape [batch_size] with dsid for each sample
        """
        # Convert to probabilities if needed
        if (abs(preds.sum(dim=-1).mean().item()-1)>0.0001):  # If logits are passed
            probs = F.softmax(preds, dim=1)
        else:
            probs = preds
        n_batch = len(targets)
        assert(n_batch <= self.max_buffer_len)
        if (self.current_update_point + n_batch) > self.max_buffer_len:
            self.current_update_point = 0 # Have to restart
            self.reset_starts()
            print("WARNING: Restarting because buffer is full")
        self.all_probs[self.current_update_point:self.current_update_point+n_batch] = probs.cpu().detach().numpy()
        self.all_targets[self.current_update_point:self.current_update_point+n_batch] = targets.cpu().detach().numpy()
        self.all_weights[self.current_update_point:self.current_update_point+n_batch] = weights.cpu().detach().numpy()
        self.all_dsids[self.current_update_point:self.current_update_point+n_batch] = dsid.cpu().to(int).detach().numpy()
        self.all_mHs[self.current_update_point:self.current_update_point+n_batch] = mH.cpu().detach().numpy()
        self.all_mWhs[self.current_update_point:self.current_update_point+n_batch] = mWh.cpu().detach().numpy()

        # Update weight sums per dsid
        unique_dsid = dsid.cpu().unique()
        for d in unique_dsid:
            if d.item() not in self.processed_weight_sums_per_dsid:
                self.processed_weight_sums_per_dsid[int(d.item())] = 0.0
            self.processed_weight_sums_per_dsid[int(d.item())] += weights[dsid == d].sum().item()
        self.current_update_point += n_batch

    def compute_and_log(self, epoch, prefix="val", step=None, log_level=0, save=True, commit=None):
        # print("Accuracy calculated: ", self.accuracy.compute())
        if log_level > -1:
            accuracies = self.compute_accuracy()
            self.starts['accuracy'] = self.current_update_point
            metrics = {
                f"{prefix}/accuracy{label}": accuracies[label] for label in accuracies.keys()
            }
            auc_scores = self.compute_auc()
            self.starts['auc'] = self.current_update_point
            metrics.update({
                f"{prefix}/auc_{label}": auc_scores[label] for label in auc_scores.keys()
            })

            metrics["epoch"] = epoch
        if log_level > 1:
            fixed_bkg_metrics = self.compute_signal_selection_metrics()
            for level_dsid, values in fixed_bkg_metrics.items():
                metrics.update({
                    # f"{prefix}_ByMassAcceptance_FixedBkg/{self.channel}_threshold/{level_dsid[0]}_{self.DSID_MASS_MAPPING[level_dsid[1]]}": values['{self.channel}_bkg_threshold'],
                    f"{prefix}_ByMassAcceptance_FixedBkg/sig_{self.channel}_expected/{level_dsid[0]}_{self.DSID_MASS_MAPPING[level_dsid[1]]}_mHlow{level_dsid[2][0]}_mHhigh{level_dsid[2][1]}": values[f'sig_{self.channel}_expected'],
                })
            self.starts['sig_sel'] = self.current_update_point
        if save:
            wandb.log({**metrics, "epoch": epoch, "step":step}, commit=commit)
        else:
            print(metrics)
        return metrics
    
    def compute_accuracy(self):
        if self.unweighted:
            raise NotImplementedError # Just needs a quick fix for the weights, probably shouldn't be done here actually
        # Convert predictions to class indices
        pred_classes = np.argmax(self.all_probs[self.starts['accuracy']:self.current_update_point], axis=1)
        # Calculate weighted correct predictions
        correct = (pred_classes == self.all_targets[self.starts['accuracy']:self.current_update_point])
        weighted_correct = correct * self.all_weights[self.starts['accuracy']:self.current_update_point]
        self.total_correct = weighted_correct.sum().item()
        self.total_weight = self.all_weights[self.starts['accuracy']:self.current_update_point].sum().item()
        if self.total_weight == 0:
            accs = {'':0}
            accs = {f'_{self.channel}':0}
        else:
            accs = {'':self.total_correct / self.total_weight}
            accs = {f'_{self.channel}':self.total_correct / self.total_weight}

        bkg_dsids = (self.all_dsids[self.starts['accuracy']:self.current_update_point] < 500000) | (self.all_dsids[self.starts['accuracy']:self.current_update_point] > 600000)
        for signal_dsid in self.DSID_MASS_MAPPING.keys():
            dsid_sel = (self.all_dsids[self.starts['accuracy']:self.current_update_point] == signal_dsid) | bkg_dsids
            pred_classes = np.argmax(self.all_probs[self.starts['accuracy']:self.current_update_point], axis=1)
            correct = (pred_classes == self.all_targets[self.starts['accuracy']:self.current_update_point]) * dsid_sel
            weighted_correct = correct * self.all_weights[self.starts['accuracy']:self.current_update_point] * dsid_sel
            total_correct = weighted_correct.sum().item()
            total_weight = (self.all_weights[self.starts['accuracy']:self.current_update_point] * dsid_sel).sum().item()
            if total_weight != 0:
                accs[f'_{self.DSID_MASS_MAPPING[signal_dsid]}'] = total_correct / total_weight
            else:
                accs[f'_{self.DSID_MASS_MAPPING[signal_dsid]}'] = 0
        return accs
    
    def compute_auc(self):
        auc_scores = {}
        # Create binary targets for this class
        binary_targets = (self.all_targets[self.starts['auc']:self.current_update_point] == 1)
        # Get probabilities for this class
        class_probs = self.all_probs[self.starts['auc']:self.current_update_point, 1]
        # Compute weighted AUC
        if len(np.unique(binary_targets)) < 2:
            return auc_scores
        # Sort by predicted probability
        sort_idx = np.argsort(class_probs)
        sort_idx = sort_idx[::-1]
        # sorted_probs = class_probs[sort_idx]
        sorted_targets = binary_targets[sort_idx]
        sorted_weights = self.all_weights[sort_idx]
        # Compute weighted TPR and FPR
        total_pos_weight = (sorted_targets * sorted_weights).sum()
        total_neg_weight = ((1 - sorted_targets) * sorted_weights).sum()
        tpr = np.cumsum(sorted_targets * sorted_weights, axis=0) / total_pos_weight
        fpr = np.cumsum((1 - sorted_targets) * sorted_weights, axis=0) / total_neg_weight
        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr).item()
        auc_scores[self.channel] = auc
        return auc_scores
    
    def compute_signal_selection_metrics(self, min_mass=0):
        assert(self.starts['sig_sel']==0) # We need to start at 0 here because otherwise the processed sums won't match
        if not len(self.all_probs):
            return {}
        
        # Separate signal and background
        bkg_mask = (self.all_targets == 0)[self.starts['sig_sel']:self.current_update_point].astype(float)
        sig_mask = (self.all_targets == 1)[self.starts['sig_sel']:self.current_update_point].astype(float)
        
        min_mass_mask = (self.all_mWhs>=min_mass)[self.starts['sig_sel']:self.current_update_point]

        # Initialize results dictionary
        results = {}

        # Set up some stuff that we only want to do once if possible
        # weight_mult_factors = np.array([self.total_weights_per_dsid[dsid.item()]/self.processed_weight_sums_per_dsid[dsid.item()] for dsid in self.all_dsids[self.starts['sig_sel']:self.current_update_point]])
        weight_mult_factors = np.array([self.total_weights_per_dsid[dsid.item()]/self.processed_weight_sums_per_dsid[dsid.item()] if self.processed_weight_sums_per_dsid[dsid.item()]!=0 else 0 for dsid in self.all_dsids[self.starts['sig_sel']:self.current_update_point]])
        # TODO Do we want to sort the vectors (only those used by, and only to be used for, the threshold calculation stuff) here? Or keep doing it all together later. Basically might help with speed
        sort_idx = np.argsort(self.all_probs[self.starts['sig_sel']:self.current_update_point, 0])
        sorted_probs_bkg = self.all_probs[self.starts['sig_sel']:self.current_update_point][sort_idx, 0]
        # sorted_weights = self.all_weights[self.starts['sig_sel']:self.current_update_point][sort_idx]

        for (mH_lower, mH_upper) in self.mH_Limits:
            mH_mask = ((self.all_mHs >= mH_lower) & (self.all_mHs <= mH_upper))[self.starts['sig_sel']:self.current_update_point].astype(float)
            for max_bkg_level in self.max_bkg_levels:
                # Calculate the threshold for lvbb background
                cum_weights = np.cumsum((self.all_weights[self.starts['sig_sel']:self.current_update_point] * weight_mult_factors * mH_mask * bkg_mask * min_mass_mask)[sort_idx], axis=0)

                if cum_weights[-1]<max_bkg_level:
                    thresh = 1.0
                else:
                    # idx = np.searchsorted(cum_weights, max_bkg_level)
                    idx = np.argmax(cum_weights>max_bkg_level) # TODO currently we are getting the first value where it's above the level; perhaps we should get the last value where it's below and then add one
                    thresh = sorted_probs_bkg[idx]
                
                above_thresh = (self.all_probs[self.starts['sig_sel']:self.current_update_point, 0] < thresh)
                for signal_dsid in self.DSID_MASS_MAPPING.keys():
                    signal_dsid_sel=(self.all_dsids == signal_dsid)[self.starts['sig_sel']:self.current_update_point].astype(float)
                    if signal_dsid in self.processed_weight_sums_per_dsid.keys():
                        # if self.processed_weight_sums_per_dsid[signal_dsid]!=0:
                        if 1:
                            weight_scale_up_factor = self.total_weights_per_dsid[signal_dsid]/self.processed_weight_sums_per_dsid[signal_dsid]
                        # else:
                        #     weight_scale_up_factor = 0
                    else:
                        weight_scale_up_factor = 0
                    selected = (above_thresh * mH_mask * min_mass_mask)

                    results[(max_bkg_level, signal_dsid, (mH_lower, mH_upper))] = {
                        f'{self.channel}_bkg_threshold': thresh,
                        f'sig_{self.channel}_expected': (selected * sig_mask * signal_dsid_sel * self.all_weights[self.starts['sig_sel']:self.current_update_point]).sum()*weight_scale_up_factor,
                    }
        return results

# Modified loss class with wandb logging
class HEPLoss(torch.nn.Module):
    def __init__(self, weight_by_mH=False, alpha=1.0, target_mass=125, apply_correlation_penalty=False):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.weight_by_mH = weight_by_mH
        self.apply_correlation_penalty = apply_correlation_penalty
        self.alpha = alpha
        self.target_mass = target_mass
        
    def forward(self, inputs, targets, weights, log, mWh, mHs):
        if self.weight_by_mH:
            weights *= self.scale_loss_by_mH(mHs)
        ce_loss = self.ce(inputs, targets) * weights
        
        bkg=targets.argmax(dim=-1)==0
        # correlation_loss = 1 - self.alpha * weighted_correlation(1-inputs[bkg,0], (inputs[bkg,1]>inputs[bkg,2])*masses_lv[bkg] + (inputs[bkg,1]<=inputs[bkg,2])*masses_qq[bkg], weights[bkg]) ** 2
        correlation_loss = weighted_correlation(1-inputs[bkg,0], mWh[bkg], weights[bkg]) ** 2


        # Mass regularization
        # qq_mass_loss = (masses_qq[targets==2] - self.target_mass).pow(2).mean()
        # lv_mass_loss = (masses_lv[targets==1] - self.target_mass).pow(2).mean()
        
        # total_loss = ce_loss.mean() + self.alpha * (qq_mass_loss + lv_mass_loss)
        if self.apply_correlation_penalty:
            total_loss = (ce_loss.sum())/(weights.sum()) + self.alpha * correlation_loss
        else:
            total_loss = (ce_loss.sum())/(weights.sum())
        
        # Log individual loss components
        if log:
            wandb.log({
                "loss/total": (ce_loss.sum())/(weights.sum()).item(),
                "loss/total_withCorrelation": (total_loss+ self.alpha * correlation_loss).item(),
                "loss/correlation": correlation_loss.item(),
                # "loss/lv_mass": lv_mass_loss.item()
            }, commit=False)
        
        return total_loss

    def scale_loss_by_mH(self, mH):
        # TODO replace this with just like a torch.gaussain or something to allow more flexibility
        TrueMh = 125e3
        maxDiff=125e3 # Can be at most 250 and min 50
        factor_at_max_diff = 0.05
        stretch_factor = np.sqrt(np.log(1/factor_at_max_diff))
        return 1/torch.exp(((mH-TrueMh)/maxDiff*stretch_factor)**2)