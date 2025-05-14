import torch 
import wandb
from utils.utils import weighted_correlation
import einops
from utils.utils import Get_PtEtaPhiM_fromXYZT, check_category

def init_wandb(config):
    wandb.init(
        project="HEP-Transformers-TruthMatchingReco",
        config=config,
        name=config["name"],
        magic=True,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

def check_valid_old(types, inclusion, padding_token, categorical):
    if categorical:
        num_in_H = {}
        num_in_W = {}
        assert(padding_token==5) # Only written this code for this; if ljets are split into 3=not-xbb, 5=xbb, then have to re-write this function
        particle_type_mapping = {0:'electron', 1:'muon', 2:'neutrino', 3:'ljet', 4:'sjet'}
        for ptype_idx in particle_type_mapping.keys():
            num_in_H[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).to(int) * (inclusion.argmax(dim=-1)==1).to(int))).sum(dim=-1)
            num_in_W[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).to(int) * (inclusion.argmax(dim=-1)==2).to(int))).sum(dim=-1)
        valid_H =   ((num_in_H['electron']==0) & (num_in_H['muon']==0) & (num_in_H['neutrino']==0) & (num_in_H['sjet']==2) & (num_in_H['ljet']==0)) | \
                    ((num_in_H['electron']==0) & (num_in_H['muon']==0) & (num_in_H['neutrino']==0) & (num_in_H['sjet']==0) & (num_in_H['ljet']==1)) 
        valid_Wlv = ((num_in_W['electron']==1) & (num_in_W['muon']==0) & (num_in_W['neutrino']==1) & (num_in_W['sjet']==0) & (num_in_W['ljet']==0)) | \
                    ((num_in_W['electron']==0) & (num_in_W['muon']==1) & (num_in_W['neutrino']==1) & (num_in_W['sjet']==0) & (num_in_W['ljet']==0)) 
        valid_Wqq = ((num_in_W['electron']==0) & (num_in_W['muon']==0) & (num_in_W['neutrino']==0) & (num_in_W['sjet']==2) & (num_in_W['ljet']==0)) | \
                    ((num_in_W['electron']==0) & (num_in_W['muon']==0) & (num_in_W['neutrino']==0) & (num_in_W['sjet']==0) & (num_in_W['ljet']==1))
        valid = valid_H & (valid_Wlv | valid_Wqq)
    else:
        num_electrons=(((types == 0).to(int) * (inclusion>0).to(int))).sum(dim=-1)
        num_muons=(((types == 1).to(int) * (inclusion>0).to(int))).sum(dim=-1)
        num_neutrinos=(((types == 2).to(int) * (inclusion>0).to(int))).sum(dim=-1)
        if padding_token==5: # all ljets are type==3
            num_ljets=(((types == 3).to(int) * (inclusion>0).to(int))).sum(dim=-1)
        elif padding_token==6: # We separated ljets into xbb type==5 and not-xbb type==3
            num_ljets=((((types == 3).to(int) * (inclusion>0).to(int))).sum(dim=-1) + (((types == 5).to(int) * inclusion)).sum(dim=-1)).to(int)
        num_sjets=(((types == 4).to(int) * (inclusion>0))).sum(dim=-1)

        valid_lvbb = ((num_electrons+num_muons)==1) & (num_neutrinos==1) & ((num_ljets==1)|(num_sjets==2))
        valid_qqbb = ((num_electrons+num_muons+num_neutrinos)==0) & ((num_ljets==2)|((num_ljets==1)&(num_sjets==2))|(num_sjets==4))
        valid = valid_lvbb | valid_qqbb
    return valid

def check_valid(types, inclusion, padding_token, categorical):
    if categorical:
        num_in_H = {}
        num_in_W = {}
        assert(padding_token == 5)  # Only written this code for this; if ljets are split into 3=not-xbb, 5=xbb, then have to re-write this function
        particle_type_mapping = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet'}
        for ptype_idx in particle_type_mapping.keys():
            num_in_H[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).to(int) * (inclusion.argmax(dim=-1) == 1).to(int))).sum(dim=-1)
            num_in_W[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).to(int) * (inclusion.argmax(dim=-1) == 2).to(int))).sum(dim=-1)
        valid_H = ((num_in_H['electron'] == 0) * (num_in_H['muon'] == 0) * (num_in_H['neutrino'] == 0) * 
                   ((num_in_H['sjet'] == 2) * (num_in_H['ljet'] == 0) + (num_in_H['sjet'] == 0) * (num_in_H['ljet'] == 1)))
        valid_Wlv = (((num_in_W['electron'] == 1) + (num_in_W['muon'] == 1)) * 
                     (num_in_W['neutrino'] == 1) * 
                     (num_in_W['sjet'] == 0) * 
                     (num_in_W['ljet'] == 0))
        valid_Wqq = ((num_in_W['electron'] == 0) * 
                     (num_in_W['muon'] == 0) * 
                     (num_in_W['neutrino'] == 0) * 
                     ((num_in_W['sjet'] == 2) * (num_in_W['ljet'] == 0) + 
                      (num_in_W['sjet'] == 0) * (num_in_W['ljet'] == 1)))
        valid = valid_H * (valid_Wlv + valid_Wqq)
    else:
        num_electrons = (((types == 0).to(int) * (inclusion > 0).to(int))).sum(dim=-1)
        num_muons = (((types == 1).to(int) * (inclusion > 0).to(int))).sum(dim=-1)
        num_neutrinos = (((types == 2).to(int) * (inclusion > 0).to(int))).sum(dim=-1)
        if padding_token == 5:  # All ljets are type==3
            num_ljets = (((types == 3).to(int) * (inclusion > 0).to(int))).sum(dim=-1)
        elif padding_token == 6:  # We separated ljets into xbb type==5 and not-xbb type==3
            num_ljets = ((((types == 3).to(int) * (inclusion > 0).to(int))).sum(dim=-1) + 
                         (((types == 5).to(int) * inclusion)).sum(dim=-1)).to(int)
        num_sjets = (((types == 4).to(int) * (inclusion > 0))).sum(dim=-1)
        valid_lvbb = ((num_electrons + num_muons == 1) * 
                      (num_neutrinos == 1) * 
                      ((num_ljets == 1) + (num_sjets == 2)))
        valid_qqbb = ((num_electrons + num_muons + num_neutrinos == 0) * 
                      ((num_ljets == 2) + 
                       ((num_ljets == 1) * (num_sjets == 2)) + 
                       (num_sjets == 4)))
        valid = valid_lvbb + valid_qqbb
    return valid

class HEPMetrics:
    def __init__(self,
                 padding_token,
                 num_objects, 
                 is_categorical=False,
                 num_categories=0,
                 max_buffer_len=100000,
                 mass_bins=50, 
                 mass_range=(0, 1000), 
                 signal_acceptance_levels=[500],
                 max_bkg_levels=[200],
                 total_weights_per_dsid={},
                 mHLimits=[(0,1e10),(95e3, 140e3)],
                 unweighted=False,
                 dsid_groups={}, # For measuring predictive power across different groups of processes
                 ):
        self.padding_token = padding_token
        self.num_objects = num_objects
        self.is_categorical = is_categorical
        if self.is_categorical:
            assert(num_categories!=0)
        self.num_categories = num_categories
        self.max_buffer_len = max_buffer_len
        assert(len(total_weights_per_dsid))
        self.total_weights_per_dsid = total_weights_per_dsid
        self.unweighted = unweighted
        self.DSID_MASS_MAPPING = {510115:0.8, 510116:0.9, 510117:1.0, 510118:1.2, 510119:1.4, 510120:1.6, 510121:1.8, 510122:2.0, 510123:2.5, 510124:3.0}
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.dsid_groups=dsid_groups
        self.reset()
        
    def reset(self):
        # Store all predictions and targets with weights
        if self.is_categorical:
            self.all_preds = torch.zeros((self.max_buffer_len, self.num_objects, self.num_categories))
        else:
            self.all_preds = torch.zeros((self.max_buffer_len, self.num_objects))
        self.all_targets = torch.zeros((self.max_buffer_len, self.num_objects))
        self.all_weights = torch.zeros(self.max_buffer_len)
        self.all_dsids = torch.zeros(self.max_buffer_len)  # to track dsid for each sample
        self.all_types = torch.zeros(self.max_buffer_len, self.num_objects) # to track whether an object is present or is to be padded for calculations
        self.processed_weight_sums_per_dsid = {}  # to track weight sums per dsid
        self.current_update_point = 0
        self.starts = {
            # 'accuracy' : 0,
            'cross_entropy':0,
            'full_reconstruction':0,
            'valid_counts':0,
        }

    def reset_starts(self, ks=None):
        if ks is None: # All of them
            self.starts = {k:0 for k in self.starts.keys()}
        else:
            for k in ks:
                self.starts[k] = 0

    def update(self, preds, targets, weights, dsid, types):
        """
        Args:
            preds: Tensor of shape [batch_size, 3] with predicted probabilities
                  (columns: background, lvbb, qqbb)
            targets: Tensor of shape [batch_size] with true class indices
            weights: Tensor of shape [batch_size] with sample weights
            dsid: Tensor of shape [batch_size] with dsid for each sample
        """
        # Convert to probabilities if needed
        # if (abs(preds.sum(dim=-1).mean().item()-1)>0.0001):  # If logits are passed
        #     # print('Softmaxing TEST REMOVE THIS PRINT STATEMENT')
        #     probs = F.softmax(preds, dim=1)
        # else:
        #     probs = preds
        n_batch = len(targets)
        assert(n_batch <= self.max_buffer_len)
        if (self.current_update_point + n_batch) > self.max_buffer_len:
            self.current_update_point = 0 # Have to restart
            self.reset_starts()
            print("WARNING: Restarting because buffer is full")
        self.all_preds[self.current_update_point:self.current_update_point+n_batch] = preds.cpu().detach()
        self.all_targets[self.current_update_point:self.current_update_point+n_batch] = targets.cpu().detach()
        self.all_weights[self.current_update_point:self.current_update_point+n_batch] = weights.cpu().detach()
        self.all_dsids[self.current_update_point:self.current_update_point+n_batch] = dsid.cpu().detach()
        self.all_types[self.current_update_point:self.current_update_point+n_batch] = types.cpu().detach() # Needed so we can mask the padding objects for metric tracking

        # Update weight sums per dsid
        unique_dsid = dsid.cpu().unique()
        for d in unique_dsid:
            if d.item() not in self.processed_weight_sums_per_dsid:
                self.processed_weight_sums_per_dsid[d.item()] = 0.0
            self.processed_weight_sums_per_dsid[d.item()] += weights[dsid == d].sum().item()
        self.current_update_point += n_batch

    def compute_and_log(self, epoch, prefix="val", step=None, log_level=0, save=True, commit=None, verbose=True, calc_all=False):
        # print("Accuracy calculated: ", self.accuracy.compute())
        if log_level > -1:
            # accuracies = self.compute_accuracy()
            # self.starts['accuracy'] = self.current_update_point
            # metrics = {
            #     f"{prefix}/accuracy{label}": accuracies[label] for label in accuracies.keys()
            # }
            metrics = {}
            losses = self.compute_loss()
            self.starts['cross_entropy'] = self.current_update_point
            metrics.update({
                f"{prefix}/loss_{label}": losses[label] for label in losses.keys()
                })
            recos = self.compute_fullrecos(calc_all=calc_all)
            self.starts['full_reconstruction'] = self.current_update_point
            for k in recos.keys():
                metrics.update({
                    f"{prefix}/{k}_{label}": recos[k][label] for label in recos[k].keys()
                    })
            valids = self.compute_valids()
            self.starts['valid_counts'] = self.current_update_point
            metrics.update({
                f"{prefix}/validPct_{label}": valids[label] for label in valids.keys()
                })
            metrics["epoch"] = epoch
        if save:
            wandb.log({**metrics, "epoch": epoch, "step":step}, commit=commit)
        elif verbose:
            print(metrics)
        return metrics
    
    def compute_valids(self):
        idx_mask = (torch.arange(len(self.all_preds))>self.starts['valid_counts']) & (torch.arange(len(self.all_preds))<self.current_update_point)
        results = {}
        mask = idx_mask
        results['all'] = (check_valid(self.all_types[mask], self.all_preds[mask], self.padding_token, self.is_categorical) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
        for dsid in self.DSID_MASS_MAPPING.keys():
            try:
                mask = idx_mask & (self.all_dsids==dsid)
                results[self.DSID_MASS_MAPPING[dsid]] = (check_valid(self.all_types[mask], self.all_preds[mask], self.padding_token, self.is_categorical) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            except:
                results[self.DSID_MASS_MAPPING[dsid]]=0
        for grp in self.dsid_groups.keys():
            try:
                mask = idx_mask & (torch.isin(self.all_dsids, self.dsid_groups[grp]))
                results[grp] = (check_valid(self.all_types[mask], self.all_preds[mask], self.padding_token, self.is_categorical) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            except:
                results[grp]=0
        return results

    def compute_fullrecos(self, calc_all=False):
        idx_mask = (torch.arange(len(self.all_preds))>self.starts['full_reconstruction']) & (torch.arange(len(self.all_preds))<self.current_update_point)
        results = {k:{} for k in ['PerfectRecoPct', 'Pct_OverPredict', 'Pct_UnderPredict', 'Pct_MisPredict']}
        mask = idx_mask
        if self.is_categorical:
            results['PerfectRecoPct']['all'] = (((self.all_preds.argmax(dim=-1)[mask]==((self.all_targets[mask]==1) + (self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
        else:
            results['PerfectRecoPct']['all'] = ((((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
        for dsid in self.DSID_MASS_MAPPING.keys():
            try:
                mask = idx_mask & (self.all_dsids==dsid)
                if self.is_categorical:
                    results['PerfectRecoPct'][self.DSID_MASS_MAPPING[dsid]] = (((self.all_preds.argmax(dim=-1)[mask]==((self.all_targets[mask]==1)+(self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
                else:
                    results['PerfectRecoPct'][self.DSID_MASS_MAPPING[dsid]] = ((((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            except:
                results['PerfectRecoPct'][self.DSID_MASS_MAPPING[dsid]]=0

            
        for channel in ['lvbb', 'qqbb']:
            if channel == 'lvbb': # Truth type of lepton/neutrino will be 3
                channel_mask = (self.all_targets==3).any(dim=-1)
            elif channel == 'qqbb': # Truth type of either small-jets or large-jet will be 2
                channel_mask = (self.all_targets==2).any(dim=-1)
            else:
                assert(False)
            mask = idx_mask & channel_mask
            if self.is_categorical:
                results['PerfectRecoPct'][f"all_{channel}"] = (((self.all_preds.argmax(dim=-1)[mask]==((self.all_targets[mask]==1) + (self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            else:
                results['PerfectRecoPct'][f"all_{channel}"] = ((((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            for dsid in self.DSID_MASS_MAPPING.keys():
                try:
                    mask = idx_mask & (self.all_dsids==dsid) & channel_mask
                    if self.is_categorical:
                        results['PerfectRecoPct'][f"{self.DSID_MASS_MAPPING[dsid]}_{channel}"] = (((self.all_preds.argmax(dim=-1)[mask]==((self.all_targets[mask]==1)+(self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
                    else:
                        results['PerfectRecoPct'][f"{self.DSID_MASS_MAPPING[dsid]}_{channel}"] = ((((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
                except:
                    results['PerfectRecoPct'][f"{self.DSID_MASS_MAPPING[dsid]}_{channel}"]=0
        
        true_reco_modes = check_category(self.all_types, self.all_targets, self.padding_token, use_torch=True)
        for reco_category in range(6): 
            # There are 6 ways to reconstruct an event correctly
            # 0: small-R-pair H, small-R-pair W
            # 1: small-R-pair H, leptonic-W
            # 2: small-R-pair H, large-R W
            # 3: large-R H, small-R-pair W
            # 4: large-R H, leptonic-W
            # 5: large-R H, large-R W
            
            category_mask = true_reco_modes==reco_category
            mask = idx_mask & category_mask
            if self.is_categorical:
                results['PerfectRecoPct'][f"all_cat{reco_category}"] = (((self.all_preds.argmax(dim=-1)[mask]==((self.all_targets[mask]==1) + (self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            else:
                results['PerfectRecoPct'][f"all_cat{reco_category}"] = ((((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            if calc_all:
                for dsid in self.DSID_MASS_MAPPING.keys():
                    try:
                        mask = idx_mask & (self.all_dsids==dsid) & category_mask
                        if self.is_categorical:
                            results['PerfectRecoPct'][f"{self.DSID_MASS_MAPPING[dsid]}_cat{reco_category}"] = (((self.all_preds.argmax(dim=-1)[mask]==((self.all_targets[mask]==1)+(self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
                        else:
                            results['PerfectRecoPct'][f"{self.DSID_MASS_MAPPING[dsid]}_cat{reco_category}"] = ((((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
                    except:
                        results['PerfectRecoPct'][f"{self.DSID_MASS_MAPPING[dsid]}_cat{reco_category}"]=0

        
        eligible_items = self.all_types!=self.padding_token
        if self.is_categorical:
            predicted_present_items = self.all_preds.argmax(dim=-1)>0
            true_present_items = self.all_targets!=0
        else:
            predicted_present_items = self.all_preds>0
            true_present_items = self.all_targets!=0


        # print("Total eligibe items:")
        # print((eligible_items).sum(dim=-1))
        # print("Total correct items:")
        # print((((predicted_present_items)==true_present_items) & (eligible_items)).sum(dim=-1)[idx_mask])
        # print("Total incorrect items:")
        # print((((predicted_present_items)!=true_present_items) & (eligible_items)).sum(dim=-1)[idx_mask])
        # print("Total items predicted:")
        # print(((predicted_present_items) & (eligible_items)).sum(dim=-1)[idx_mask])
        # print("Total items truth:")
        # print((true_present_items & (eligible_items)).sum(dim=-1)[idx_mask])
        # print("Total over-prediction (pred-total):")
        # print((((predicted_present_items) & (eligible_items)).sum(dim=-1) - (true_present_items & (eligible_items)).sum(dim=-1))[idx_mask])
        over_prediction_amount = (predicted_present_items & eligible_items).sum(dim=-1) - (true_present_items & eligible_items).sum(dim=-1)
        



        # print("Perecentage under-predicted")
        # print(((over_prediction_amount<0)*self.all_weights)[idx_mask].sum()/self.all_weights[idx_mask].sum())
        mask = idx_mask
        results['Pct_UnderPredict']['all'] = ((over_prediction_amount<0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
        for channel in ['lvbb', 'qqbb']:
            if channel == 'lvbb': # Truth type of lepton/neutrino will be 3
                channel_mask = (self.all_targets==3).any(dim=-1)
            elif channel == 'qqbb': # Truth type of either small-jets or large-jet will be 2
                channel_mask = (self.all_targets==2).any(dim=-1)
            else:
                assert(False)
            mask = idx_mask & channel_mask
            try:
                results['Pct_UnderPredict'][f"all_{channel}"] = ((over_prediction_amount<0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
            except:
                results['Pct_UnderPredict'][f"all_{channel}"]=0
        if calc_all:
            for dsid in self.DSID_MASS_MAPPING.keys():
                try:
                    mask = idx_mask & (self.all_dsids==dsid)
                    results['Pct_UnderPredict'][self.DSID_MASS_MAPPING[dsid]] = ((over_prediction_amount<0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
                except:
                    results['Pct_UnderPredict'][self.DSID_MASS_MAPPING[dsid]]=0
            true_reco_modes = check_category(self.all_types, self.all_targets, self.padding_token, use_torch=True)
            for reco_category in range(6): 
                category_mask = true_reco_modes==reco_category
                try:
                    mask = idx_mask & category_mask
                    results['Pct_UnderPredict'][f"all_cat{reco_category}"] = ((over_prediction_amount<0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
                except:
                    results['Pct_UnderPredict'][f"all_cat{reco_category}"] = 0


        # print("Perecentage over-predicted")
        # print(((over_prediction_amount>0)*self.all_weights)[idx_mask].sum()/self.all_weights[idx_mask].sum())
        mask = idx_mask
        results['Pct_OverPredict']['all'] = ((over_prediction_amount>0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
        for channel in ['lvbb', 'qqbb']:
            if channel == 'lvbb': # Truth type of lepton/neutrino will be 3
                channel_mask = (self.all_targets==3).any(dim=-1)
            elif channel == 'qqbb': # Truth type of either small-jets or large-jet will be 2
                channel_mask = (self.all_targets==2).any(dim=-1)
            else:
                assert(False)
            mask = idx_mask & channel_mask
            try:
                results['Pct_OverPredict'][f"all_{channel}"] = ((over_prediction_amount>0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
            except:
                results['Pct_OverPredict'][f"all_{channel}"]=0
        if calc_all:
            for dsid in self.DSID_MASS_MAPPING.keys():
                try:
                    mask = idx_mask & (self.all_dsids==dsid)
                    results['Pct_OverPredict'][self.DSID_MASS_MAPPING[dsid]] = ((over_prediction_amount>0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
                except:
                    results['Pct_OverPredict'][self.DSID_MASS_MAPPING[dsid]]=0
            true_reco_modes = check_category(self.all_types, self.all_targets, self.padding_token, use_torch=True)
            for reco_category in range(6): 
                category_mask = true_reco_modes==reco_category
                try:
                    mask = idx_mask & category_mask
                    results['Pct_OverPredict'][f"all_cat{reco_category}"] = ((over_prediction_amount>0)*self.all_weights)[mask].sum()/self.all_weights[mask].sum()
                except:
                    results['Pct_OverPredict'][f"all_cat{reco_category}"] = 0


        # print("Percentage predicted correct # but wrong")
        # print((((over_prediction_amount==0)&(~(((predicted_present_items==true_present_items)|(~eligible_items)).all(dim=-1))))*self.all_weights)[idx_mask].sum()/self.all_weights[idx_mask].sum())
        mask = idx_mask
        if self.is_categorical:
            results['Pct_MisPredict']['all'] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask].argmax(dim=-1)==((self.all_targets[mask]==1)+(self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1))))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
        else:
            results['Pct_MisPredict']['all'] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1)))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
        for channel in ['lvbb', 'qqbb']:
            if channel == 'lvbb': # Truth type of lepton/neutrino will be 3
                channel_mask = (self.all_targets==3).any(dim=-1)
            elif channel == 'qqbb': # Truth type of either small-jets or large-jet will be 2
                channel_mask = (self.all_targets==2).any(dim=-1)
            else:
                assert(False)
            mask = idx_mask & channel_mask
            try:
                if self.is_categorical:
                    results['Pct_MisPredict'][f"all_{channel}"] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask].argmax(dim=-1)==((self.all_targets[mask]==1)+(self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1))))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
                else:
                    results['Pct_MisPredict'][f"all_{channel}"] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1)))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
            except:
                results['Pct_OverPredict'][f"all_{channel}"]=0
        if calc_all:
            for dsid in self.DSID_MASS_MAPPING.keys():
                try:
                    mask = idx_mask & (self.all_dsids==dsid)
                    if self.is_categorical:
                        results['Pct_MisPredict'][self.DSID_MASS_MAPPING[dsid]] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask].argmax(dim=-1)==((self.all_targets[mask]==1)+(self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1))))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
                    else:
                        results['Pct_MisPredict'][self.DSID_MASS_MAPPING[dsid]] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1)))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
                except:
                    results['Pct_MisPredict'][self.DSID_MASS_MAPPING[dsid]]=0
            true_reco_modes = check_category(self.all_types, self.all_targets, self.padding_token, use_torch=True)
            for reco_category in range(6): 
                category_mask = true_reco_modes==reco_category
                try:
                    mask = idx_mask & category_mask
                    if self.is_categorical:
                        results['Pct_MisPredict'][f"all_cat{reco_category}"] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask].argmax(dim=-1)==((self.all_targets[mask]==1)+(self.all_targets[mask]>1)*2))|(self.all_types[mask]==self.padding_token)).all(dim=-1))))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
                    else:
                        results['Pct_MisPredict'][f"all_cat{reco_category}"] = (((over_prediction_amount[mask]==0)&(~(((self.all_preds[mask]>0)==self.all_targets[mask].to(bool))|(self.all_types[mask]==self.padding_token)).all(dim=-1)))*self.all_weights[mask]).sum()/self.all_weights[mask].sum()
                except:
                    results['Pct_MisPredict'][f"all_cat{reco_category}"] = 0


        # print("Percentage predicted correctly")
        # print(((((predicted_present_items==true_present_items)|(~eligible_items)).all(dim=-1))*self.all_weights)[idx_mask].sum()/self.all_weights[idx_mask].sum())

        return results
    
    def compute_loss(self):
        return {}
        idx_mask = (torch.arange(len(self.all_preds))>self.starts['cross_entropy']) & (torch.arange(len(self.all_preds))<self.current_update_point)
        results = {'all':(self.loss(self.all_preds[idx_mask], self.all_targets[idx_mask]) * self.all_weights[idx_mask]).sum() / self.all_weights[idx_mask].sum()}
        for dsid in self.DSID_MASS_MAPPING.keys():
            try:
                mask = idx_mask & (self.all_dsids==dsid)
                results[self.DSID_MASS_MAPPING[dsid]] = (self.loss(self.all_preds[mask], self.all_targets[mask]) * self.all_weights[mask]).sum() / self.all_weights[mask].sum()
            except:
                results[self.DSID_MASS_MAPPING[dsid]]=0
        return results

# Modified loss class with wandb logging
class HEPLoss(torch.nn.Module):
    def __init__(self, is_categorical=False, weight_by_mH=False, alpha=1.0, target_mass=125, apply_correlation_penalty=False, apply_valid_penalty=False, valid_penalty_weight=1.0):
        super().__init__()
        self.is_categorical = is_categorical
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.weight_by_mH = weight_by_mH
        self.alpha = alpha
        self.apply_correlation_penalty = apply_correlation_penalty
        self.apply_valid_penalty = apply_valid_penalty
        self.valid_penalty_weight = valid_penalty_weight
        
    def forward(self, inputs, targets, types, padding_token, n_objs, weights, log, x_inputs=None):
        # if self.weight_by_mH:
        #     weights *= self.scale_loss_by_mH(mHs)
        
        if 0: # Loss all together
            ce_loss = self.ce(inputs, targets) * weights
        else: # Loss per object
            if self.is_categorical:
                simplified_true_inclusion = ((targets==1)*1 + (targets>1)*2) # 1 is Higgs, 2/3 is W for qqbb/lvbb cases
                one_hot_true_inclusion = torch.nn.functional.one_hot(simplified_true_inclusion.to(torch.long))
                flattened_ce = self.ce(einops.rearrange(inputs, 'batch object cls -> (batch object) cls'),
                                       einops.rearrange(one_hot_true_inclusion.float(), 'batch object cls -> (batch object) cls'),
                                       )
                num_nonempty_objs = (types!=padding_token).sum(dim=-1)
                ce_loss = flattened_ce * (einops.repeat((weights/num_nonempty_objs), 'batch -> batch max_n_objects',max_n_objects=n_objs)*(types!=padding_token)).flatten()
            else:
                num_nonempty_objs = (types!=padding_token).sum(dim=-1)
                ce_loss = self.bce(inputs.flatten(), (targets!=0).to(float).flatten()) * (einops.repeat((weights/num_nonempty_objs), 'batch -> batch max_n_objects',max_n_objects=n_objs)*(types!=padding_token)).flatten()
        
        isvalid_loss = 1 - (check_valid(types, inputs, padding_token, self.is_categorical) * weights).sum() / weights.sum()

        
        if self.is_categorical:
            assert(x_inputs is not None)
            unflattened_ce = (einops.rearrange(flattened_ce, '(batch object) -> batch object', batch=len(targets))*(types!=padding_token)).sum(dim=-1)
            pred_inclusions = torch.argmax(inputs, dim=-1)
            # corrects = (simplified_true_inclusion == pred_inclusions)
            # all_corrects = corrects.all(dim=-1)
            if 0:
                fourmom = x_inputs[...,:4] * pred_inclusions.unsqueeze(-1)
            else:
                fourmom = x_inputs[...,:4] * targets.unsqueeze(-1)
            # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,0].sum(dim=-1).cpu().numpy(),fourmom[:,1].sum(dim=-1).cpu().numpy(),fourmom[:,2].sum(dim=-1).cpu().numpy(),fourmom[:,3].sum(dim=-1).cpu().numpy())
            # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(dim=-1).cpu().numpy(),fourmom[...,1].sum(dim=-1).cpu().numpy(),fourmom[...,2].sum(dim=-1).cpu().numpy(),fourmom[...,3].sum(dim=-1).cpu().numpy())
            _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,:,0].sum(axis=-1),fourmom[:,:,1].sum(axis=-1),fourmom[:,:,2].sum(axis=-1),fourmom[:,:,3].sum(axis=-1), use_torch=True)
            correlation_loss = weighted_correlation(unflattened_ce, m, weights) ** 2
        else:
            if self.apply_correlation_penalty:
                raise NotImplementedError
            else:
                pass
        # correlation_loss = weighted_correlation(ce_loss, (inputs[bkg,1]>inputs[bkg,2])*masses_lv[bkg] + (inputs[bkg,1]<=inputs[bkg,2])*masses_qq[bkg], weights[bkg]) ** 2
        
        total_loss = (ce_loss.sum())/(weights.sum())
        if self.apply_correlation_penalty:
            total_loss += self.alpha * correlation_loss
        if self.apply_valid_penalty:
            total_loss += self.valid_penalty_weight * isvalid_loss

        # Log individual loss components
        if log:
            wandb.log({
                "loss/ce": (ce_loss.sum())/(weights.sum()).item(),
                "loss/total_withCorrelation": ((ce_loss.sum())/(weights.sum()) + self.alpha * correlation_loss).item(),
                "loss/total_withCorrelationAndValid": ((ce_loss.sum())/(weights.sum()) + self.alpha * correlation_loss + self.valid_penalty_weight * isvalid_loss).item(),
                "loss/correlation": correlation_loss.item(),
                "loss/valid": isvalid_loss.item(),
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







# Modified loss class with wandb logging
class HEPLossWithEntropy(torch.nn.Module):
    def __init__(self, is_categorical=False, weight_by_mH=False, alpha=1.0, target_mass=125, apply_correlation_penalty=False, apply_valid_penalty=False, valid_penalty_weight=1.0, entropy_loss=False, entropy_weight=1.0, target_entropy=0.0):
        super().__init__()
        self.is_categorical = is_categorical
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.weight_by_mH = weight_by_mH
        self.alpha = alpha
        self.apply_correlation_penalty = apply_correlation_penalty
        self.apply_valid_penalty = apply_valid_penalty
        self.valid_penalty_weight = valid_penalty_weight
        self.entropy_loss = entropy_loss
        self.entropy_weight = entropy_weight
        self.target_entropy = target_entropy
        
    def forward(self, cache, inputs, targets, types, padding_token, n_objs, weights, log, x_inputs=None):
        # if self.weight_by_mH:
        #     weights *= self.scale_loss_by_mH(mHs)
        
        if 0: # Loss all together
            ce_loss = self.ce(inputs, targets) * weights
        else: # Loss per object
            if self.is_categorical:
                simplified_true_inclusion = ((targets==1)*1 + (targets>1)*2) # 1 is Higgs, 2/3 is W for qqbb/lvbb cases
                one_hot_true_inclusion = torch.nn.functional.one_hot(simplified_true_inclusion.to(torch.long))
                flattened_ce = self.ce(einops.rearrange(inputs, 'batch object cls -> (batch object) cls'),
                                       einops.rearrange(one_hot_true_inclusion.float(), 'batch object cls -> (batch object) cls'),
                                       )
                num_nonempty_objs = (types!=padding_token).sum(dim=-1)
                ce_loss = flattened_ce * (einops.repeat((weights/num_nonempty_objs), 'batch -> batch max_n_objects',max_n_objects=n_objs)*(types!=padding_token)).flatten()
            else:
                num_nonempty_objs = (types!=padding_token).sum(dim=-1)
                ce_loss = self.bce(inputs.flatten(), (targets!=0).to(float).flatten()) * (einops.repeat((weights/num_nonempty_objs), 'batch -> batch max_n_objects',max_n_objects=n_objs)*(types!=padding_token)).flatten()
        
        isvalid_loss = 1 - (check_valid(types, inputs, padding_token, self.is_categorical) * weights).sum() / weights.sum()
        
        
        
        total_entropy_loss = 0
        total_heads = 0
        eps=1e-8
        entropy_losses = {}
        for layer in range(len([k for k in cache.store.keys() if (('attention' in k) and (not ('post' in k)))])):
            entropy_losses[layer] = {}
            for head in range(cache.store[f'block_{layer}_attention']['attn_weights_per_head'].shape[1]):
                attn_wts = (cache[f'block_{layer}_attention']['attn_weights_per_head'][:,head,...]) # Shape [batch object_query object_key]
                # Calculate the entropy along the last dimension (note, the padding should have been handled already so they should be 0)
                entropy_l = - (attn_wts * (attn_wts + eps).log()).sum(dim=-1)
                # entropy_l = (entropy_l - self.target_entropy).abs()
                entropy_losses[layer][head] = einops.einsum(entropy_l * (types!=padding_token), 'batch object -> batch') / einops.einsum(types!=padding_token, 'batch object -> batch')
                # total_entropy_loss += (entropy_losses[layer][head] * weights).sum() / weights.sum()
                total_entropy_loss += (entropy_losses[layer][head]).mean()
                total_heads += 1
        
        
        if self.is_categorical:
            assert(x_inputs is not None)
            unflattened_ce = (einops.rearrange(flattened_ce, '(batch object) -> batch object', batch=len(targets))*(types!=padding_token)).sum(dim=-1)
            pred_inclusions = torch.argmax(inputs, dim=-1)
            # corrects = (simplified_true_inclusion == pred_inclusions)
            # all_corrects = corrects.all(dim=-1)
            if 0:
                fourmom = x_inputs[...,:4] * pred_inclusions.unsqueeze(-1)
            else:
                fourmom = x_inputs[...,:4] * targets.unsqueeze(-1)
            # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,0].sum(dim=-1).cpu().numpy(),fourmom[:,1].sum(dim=-1).cpu().numpy(),fourmom[:,2].sum(dim=-1).cpu().numpy(),fourmom[:,3].sum(dim=-1).cpu().numpy())
            # _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[...,0].sum(dim=-1).cpu().numpy(),fourmom[...,1].sum(dim=-1).cpu().numpy(),fourmom[...,2].sum(dim=-1).cpu().numpy(),fourmom[...,3].sum(dim=-1).cpu().numpy())
            _,_,_,m=Get_PtEtaPhiM_fromXYZT(fourmom[:,:,0].sum(axis=-1),fourmom[:,:,1].sum(axis=-1),fourmom[:,:,2].sum(axis=-1),fourmom[:,:,3].sum(axis=-1), use_torch=True)
            correlation_loss = weighted_correlation(unflattened_ce, m, weights) ** 2
        else:
            if self.apply_correlation_penalty:
                raise NotImplementedError
            else:
                pass
        # correlation_loss = weighted_correlation(ce_loss, (inputs[bkg,1]>inputs[bkg,2])*masses_lv[bkg] + (inputs[bkg,1]<=inputs[bkg,2])*masses_qq[bkg], weights[bkg]) ** 2
        
        total_loss = (ce_loss.sum())/(weights.sum())
        if self.apply_correlation_penalty:
            total_loss += self.alpha * correlation_loss
        if self.apply_valid_penalty:
            total_loss += self.valid_penalty_weight * isvalid_loss
        if self.entropy_loss:
            total_loss += self.entropy_weight * (total_entropy_loss/total_heads)
        
        # Log individual loss components
        if log:
            wandb.log({
                "loss/ce": (ce_loss.sum())/(weights.sum()).item(),
                "loss/total_withCorrelation": ((ce_loss.sum())/(weights.sum()) + self.alpha * correlation_loss).item(),
                "loss/total_withCorrelationAndValid": ((ce_loss.sum())/(weights.sum()) + self.alpha * correlation_loss + self.valid_penalty_weight * isvalid_loss).item(),
                "loss/correlation": correlation_loss.item(),
                "loss/valid": isvalid_loss.item(),
                "loss/entropy": total_entropy_loss.item(),
                # "loss/lv_mass": lv_mass_loss.item()
            }, commit=False)
            wandb.log({
                f"loss/z_entropy_layer_{layer}_head_{head}": entropy_losses[layer][head].mean().item() for layer in entropy_losses.keys() for head in entropy_losses[layer].keys()
            }, commit=False)
            return total_loss
        else:
            loss_dict = {
                "loss/ce": (ce_loss.sum())/(weights.sum()).item(),
                "loss/total_withCorrelation": ((ce_loss.sum())/(weights.sum()) + self.alpha * correlation_loss).item(),
                "loss/total_withCorrelationAndValid": ((ce_loss.sum())/(weights.sum()) + self.alpha * correlation_loss + self.valid_penalty_weight * isvalid_loss).item(),
                "loss/correlation": correlation_loss.item(),
                "loss/valid": isvalid_loss.item(),
                "loss/entropy": total_entropy_loss.item(),
                # f"loss/z_entropy_layer_{layer}_head_{head}": entropy_losses[layer][head].mean().item() for layer in entropy_losses.keys() for head in entropy_losses[layer].keys()
            }
            loss_dict.update({f"loss/z_entropy_layer_{layer}_head_{head}": entropy_losses[layer][head].mean().item() for layer in entropy_losses.keys() for head in entropy_losses[layer].keys()})
            return total_loss, loss_dict
        

    def scale_loss_by_mH(self, mH):
        # TODO replace this with just like a torch.gaussain or something to allow more flexibility
        TrueMh = 125e3
        maxDiff=125e3 # Can be at most 250 and min 50
        factor_at_max_diff = 0.05
        stretch_factor = np.sqrt(np.log(1/factor_at_max_diff))
        return 1/torch.exp(((mH-TrueMh)/maxDiff*stretch_factor)**2)