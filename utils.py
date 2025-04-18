import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from typing import List
import matplotlib as mpl
import einops

import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()


def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
        event_level_features=[],
        labels=['dsid', 'truth_decay_mode'],
        new_inputs_labels=False):
    """Loads a single file from the JetClass dataset.

    **Arguments**

    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet. 
        Jets with fewer particles will be zero-padded, 
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded.
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded. 
    - **labels** : _List[str]_
        - A list of truth labels to be loaded. 

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features 
                         in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
                    in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth lables
               in the shape `(num_jets, num_classes)`.
    """

    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    if new_inputs_labels:
        table = uproot.open(filepath)['tree'].arrays()
        p4 = vector.zip({'px': table['ll_particle_px'],
                            'py': table['ll_particle_py'],
                            'pz': table['ll_particle_pz'],
                            'energy': table['ll_particle_e']})
        # pos = labels.index('dsid')
        # labels.remove('dsid')
        # labels.insert(pos, "DSID")
        # pos = labels.index('truth_decay_mode')
        # labels.remove('truth_decay_mode')
        # labels.insert(pos, "ll_truth_decay_mode")
    else:
        table = uproot.open(filepath)['ProcessedTree'].arrays()
        p4 = vector.zip({'px': table['px'],
                            'py': table['py'],
                            'pz': table['pz'],
                            'energy': table['e']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi
    table['part_px'] = p4.px
    table['part_py'] = p4.py
    table['part_pz'] = p4.pz
    table['part_energy'] = p4.energy
    table['part_mass'] = p4.mass
    if len(particle_features):
        # print(particle_features)
        x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    else:
        x_particles = None
    if len(event_level_features):
        x_event = np.stack([ak.to_numpy(table[n]).astype('float32') for n in event_level_features], axis=1)
    else:
        x_event = None
    y = np.stack([(ak.to_numpy(table[n])=='lvbb')*1+(ak.to_numpy(table[n])=='qqbb')*2 if n=='truth_W_decay_mode' else ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)

    return x_particles, x_event, y


# %%
def print_inclusion(inclusion, types, print_not_included=True):
    '''
    returns the the inclusion string of an event, for nicely printing a single event for eg. debugging or mech interp
    # inclusion shape [object]
    # types shape [object]
    '''
    type_short_mapping = {0:'e', 1:'m', 2:'n', 3:'J', 4:'j'}
    incl_str = ''
    for obj in range(len(inclusion)):
        if types[obj]!=5:
            if inclusion[obj] == 0:
                if print_not_included:
                    incl_str += ' -'
                    incl_str += type_short_mapping[types[obj].item()]
                    incl_str += ' '
                else:
                    incl_str += ' -- '
            elif inclusion[obj] == 1:
                incl_str += ' H'
                incl_str += type_short_mapping[types[obj].item()]
                incl_str += ' '
            elif inclusion[obj] >= 2:
                incl_str += ' W'
                incl_str += type_short_mapping[types[obj].item()]
                incl_str += ' '
    return incl_str

def print_vars(infotensor, types):
    '''
    # Expects infosenor to have shape [1 objects 5] and contain px, py, pz, E, tagInfo
    #       types to have shape [1 objects]
    '''
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'ljetXbbTagged'}
    types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5:'None'}
    infotensor = torch.stack((*Get_PtEtaPhiM_fromXYZT(infotensor[...,0],infotensor[...,1],infotensor[...,2],infotensor[...,3],use_torch=True), infotensor[...,4]),dim=-1)[0]
    types = types[0]
    print(f'        {"Type":15s}, {"Pt":6s}, {"Eta":6s}, {"Phi":6s}, {"M":6s}')
    for row in range(infotensor.shape[-2]):
        if types_dict[types[row].item()]=='None':
            continue
        print(f"Object: {types_dict[types[row].item()]:10s}, ", end=" ")
        for col in range(infotensor.shape[-1]):
            print(f"{infotensor[row][col]:6.3f}, ", end='')
        print('')

# %%
    

def check_valid(types, inclusion, padding_token, categorical, returnTypes = False):
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
        if returnTypes:
            return valid, (valid_H & valid_Wlv).to(int) + (valid_H & valid_Wqq).to(int)*2
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
        if returnTypes:
            return valid, valid_lvbb.to(int) + valid_qqbb.to(int)*2
    return valid


def check_category(types, #[batch object]
                   inclusion, # [batch object]
                   padding_token, # int
                   use_torch=False, # bool
                   ):
    num_in_H = {}
    num_in_W = {}
    assert(padding_token==5) # Only written this code for this; if ljets are split into 3=not-xbb, 5=xbb, then have to re-write this function
    particle_type_mapping = {0:'electron', 1:'muon', 2:'neutrino', 3:'ljet', 4:'sjet'}
    if not use_torch:
        category = np.ones(len(types), dtype=int)*-1
        for ptype_idx in particle_type_mapping.keys():
            num_in_H[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).astype(int) * (inclusion==1).astype(int))).sum(axis=-1)
            num_in_W[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).astype(int) * (inclusion>1).astype(int))).sum(axis=-1)
    else:
        category = torch.ones(len(types), dtype=torch.int)*-1
        for ptype_idx in particle_type_mapping.keys():
            num_in_H[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).to(int) * (inclusion==1).to(int))).sum(dim=-1)
            num_in_W[particle_type_mapping[ptype_idx]] = (((types == ptype_idx).to(int) * (inclusion>1).to(int))).sum(dim=-1)
    valid_H_sjet =   ((num_in_H['electron']==0) & (num_in_H['muon']==0) & (num_in_H['neutrino']==0) & (num_in_H['sjet']==2) & (num_in_H['ljet']==0))
    valid_H_ljet =   ((num_in_H['electron']==0) & (num_in_H['muon']==0) & (num_in_H['neutrino']==0) & (num_in_H['sjet']==0) & (num_in_H['ljet']==1)) 
    valid_W_lv = ((num_in_W['electron']==1) & (num_in_W['muon']==0) & (num_in_W['neutrino']==1) & (num_in_W['sjet']==0) & (num_in_W['ljet']==0)) | \
                ((num_in_W['electron']==0) & (num_in_W['muon']==1) & (num_in_W['neutrino']==1) & (num_in_W['sjet']==0) & (num_in_W['ljet']==0)) 
    valid_W_sjet = ((num_in_W['electron']==0) & (num_in_W['muon']==0) & (num_in_W['neutrino']==0) & (num_in_W['sjet']==2) & (num_in_W['ljet']==0))
    valid_W_ljet = ((num_in_W['electron']==0) & (num_in_W['muon']==0) & (num_in_W['neutrino']==0) & (num_in_W['sjet']==0) & (num_in_W['ljet']==1))
    category[valid_H_sjet & valid_W_sjet]   = 0
    category[valid_H_sjet & valid_W_lv]     = 1
    category[valid_H_sjet & valid_W_ljet]   = 2
    category[valid_H_ljet & valid_W_sjet]   = 3
    category[valid_H_ljet & valid_W_lv]     = 4
    category[valid_H_ljet & valid_W_ljet]   = 5
    return category





# %%
# Cosine learning rate scheduler parameters
def cosine_lr_scheduler(epoch: int, lr_high: float, lr_low: float, n_epochs: int):
    """
    This function calculates the learning rate following a cosine schedule
    that oscillates between `lr_high` and `lr_low` every `n_epochs`.
    """
    # Cosine annealing function
    if epoch<n_epochs:
        cycle_epoch = epoch % n_epochs
        progress = cycle_epoch / n_epochs
        lr = lr_low + 0.5 * (lr_high - lr_low) * (1 + math.cos(math.pi * progress))
    else:
        lr = lr_low
    return lr

def multi_cosine_lr_scheduler(epoch: int, lrs:List[float], n_epochs: int):
    """
    This function calculates the learning rate following a cosine schedule
    flattens every (n_epochs)/len(lrs) epochs, at each element of lrs
    """
    # Cosine annealing function
    num_epochs_per_chunk = int(n_epochs/(len(lrs)-1))
    chunk = epoch // num_epochs_per_chunk
    if chunk<len(lrs)-1:
        lr_high = lrs[chunk]
        lr_low = lrs[chunk+1]
        cycle_epoch = epoch % num_epochs_per_chunk
        progress = cycle_epoch / num_epochs_per_chunk
        lr = lr_low + 0.5 * (lr_high - lr_low) * (1 + math.cos(math.pi * progress))
    else:
        lr = lrs[-1]
    return lr

def basic_lr_scheduler(epoch: int, lr_high: float, lr_low: float, n_epochs: int, log=True, warmup_steps=None, warmup_rate=None):
    """
    This function calculates the learning rate following a flat decreasing schedule
    """
    if (warmup_steps is not None) and (warmup_rate is not None):
        if epoch<warmup_steps:
            return warmup_rate
    if log:
        return lr_high * np.power((lr_low/lr_high), epoch/n_epochs)
    else:
        return lr_high - (lr_high - lr_low)*(epoch/n_epochs)

# %%
def myhist2d(xvals, 
             yvals,
             wvals=None,
             logx = True,
             logy = True,
             logz = False,
             nbins=100,
             xlabel='',
             ylabel='',
):
    if (wvals is None):
        wvals = np.ones_like(xvals)
    bins=[nbins,nbins]
    if logx:
        log_binsx = np.logspace(np.log10(min(xvals).item()), np.log10(max(xvals).item()), num=nbins)
        bins[0] = log_binsx
    if logy:
        log_binsy = np.logspace(np.log10(min(yvals).item()), np.log10(max(yvals).item()), num=nbins)
        bins[1] = log_binsy
    corrcoeff = weighted_correlation(torch.from_numpy(xvals), torch.from_numpy(yvals), torch.from_numpy(wvals)).item()
    plt.figure(figsize=(4.5,4))
    if logz:
        plt.hist2d(xvals, yvals, weights=wvals, norm=mpl.colors.LogNorm(), bins=bins)
    else:
        plt.hist2d(xvals, yvals, weights=wvals, bins=bins)
    # plt.hist2d(xvals, yvals, weights=wvals, bins=[log_binsx, log_binsy])
    plt.title(f'Weighted by abs(MC)\nCorrelation: {corrcoeff:5f}')
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# %%

def weighted_correlation(x, y, w):
    # Ensure that the tensors are 1D and have the same length
    assert x.ndimension() == 1 and y.ndimension() == 1 and w.ndimension() == 1, "All inputs must be 1D tensors"
    assert len(x) == len(y) == len(w), "x, y, and w must have the same length"
    
    # Normalize the weights
    w_sum = torch.sum(w)
    
    # Compute weighted means
    weighted_mean_x = torch.sum(x * w) / w_sum
    weighted_mean_y = torch.sum(y * w) / w_sum
    
    # Compute weighted covariance
    weighted_cov_xy = torch.sum(w * (x - weighted_mean_x) * (y - weighted_mean_y)) / w_sum
    
    # Compute weighted variances
    weighted_var_x = torch.sum(w * (x - weighted_mean_x)**2) / w_sum
    weighted_var_y = torch.sum(w * (y - weighted_mean_y)**2) / w_sum
    
    # Compute weighted correlation
    correlation = weighted_cov_xy / (torch.sqrt(weighted_var_x * weighted_var_y))
    
    return correlation


# %%
# Define useful transforms for when we're reading the data in
def Get_PtEtaPhiM_fromXYZT(obj_px, obj_py, obj_pz, obj_e, use_torch=False):
    '''
    Takes in arrays of shape (n_batch[,n_obj],) for x, y, z and t (==e) 
    of some objects and returns arrays of shape (n_batch[,n_obj],) containing 
    the Pt, Eta, Phi and M of the objects. Each element of n_batch corresponds
    to one event, and each of the n_objs represents an object in the event.
    '''
    if use_torch:
        obj_pt = torch.sqrt((obj_px ** 2 + obj_py**2))
        obj_ptot = torch.sqrt((obj_px ** 2 + obj_py**2 + obj_pz**2)) # sqrt*(x^2 + y^2 + z^2) == rho in spherical coords
        
        obj_cosTheta = torch.empty_like(obj_px)
        obj_cosTheta[obj_ptot==0] = 1
        obj_cosTheta[obj_ptot!=0] = obj_pz[obj_ptot!=0]/obj_ptot[obj_ptot!=0]

        obj_Eta = torch.empty_like(obj_px)
        eta_valid_mask = obj_cosTheta*obj_cosTheta < 1
        obj_Eta[eta_valid_mask] = -0.5* torch.log( (1.0-obj_cosTheta[eta_valid_mask])/(1.0+obj_cosTheta[eta_valid_mask]) )
        obj_Eta[(~eta_valid_mask) & (obj_pz==0)] = 0
        obj_Eta[(~eta_valid_mask) & (obj_pz>0)] = torch.inf
        obj_Eta[(~eta_valid_mask) & (obj_pz<0)] = -torch.inf
        
        obj_Phi = torch.empty_like(obj_px)
        phi_valid_mask = ~((obj_px == 0) & (obj_py == 0))
        obj_Phi[phi_valid_mask] = torch.atan2(obj_py[phi_valid_mask], obj_px[phi_valid_mask])
        obj_Phi[~phi_valid_mask] = 0

        obj_Mag2 = obj_e**2 - obj_ptot**2

        obj_M = torch.empty_like(obj_px)
        obj_M[obj_Mag2<0] = -torch.sqrt((-obj_Mag2[obj_Mag2<0]))
        obj_M[obj_Mag2>=0] = torch.sqrt((obj_Mag2[obj_Mag2>=0]))

        return obj_pt, obj_Eta, obj_Phi, obj_M
    else:
        obj_pt = np.sqrt((obj_px ** 2 + obj_py**2))
        obj_ptot = np.sqrt((obj_px ** 2 + obj_py**2 + obj_pz**2)) # sqrt*(x^2 + y^2 + z^2) == rho in spherical coords
        
        obj_cosTheta = np.empty_like(obj_px)
        obj_cosTheta[obj_ptot==0] = 1
        obj_cosTheta[obj_ptot!=0] = obj_pz[obj_ptot!=0]/obj_ptot[obj_ptot!=0]

        obj_Eta = np.empty_like(obj_px)
        eta_valid_mask = obj_cosTheta*obj_cosTheta < 1
        obj_Eta[eta_valid_mask] = -0.5* np.log( (1.0-obj_cosTheta[eta_valid_mask])/(1.0+obj_cosTheta[eta_valid_mask]) )
        obj_Eta[(~eta_valid_mask) & (obj_pz==0)] = 0
        obj_Eta[(~eta_valid_mask) & (obj_pz>0)] = np.inf
        obj_Eta[(~eta_valid_mask) & (obj_pz<0)] = -np.inf
        
        obj_Phi = np.empty_like(obj_px)
        phi_valid_mask = ~((obj_px == 0) & (obj_py == 0))
        obj_Phi[phi_valid_mask] = np.atan2(obj_py[phi_valid_mask], obj_px[phi_valid_mask])
        obj_Phi[~phi_valid_mask] = 0

        obj_Mag2 = obj_e**2 - obj_ptot**2

        obj_M = np.empty_like(obj_px)
        obj_M[obj_Mag2<0] = -np.sqrt((-obj_Mag2[obj_Mag2<0]))
        obj_M[obj_Mag2>=0] = np.sqrt((obj_Mag2[obj_Mag2>=0]))

        return obj_pt, obj_Eta, obj_Phi, obj_M


def get_num_btags(x_part, types, pred_inclusion, tag_info_idx, use_torch=False): # Put it down here because it needs the Get_PtEtaPhiM_fromXYZT function
    # Function to get the number of b-tagged small-R jets (sjets) passing a given working point
    # x_part: (n_batch, n_obj, n_features) tensor of particle features
    # types: (n_batch, n_obj) tensor of particle types. 0=electron, 1=muon, 2=neutrino, 3=ljet, 4=sjet
    # pred_inclusion: (n_batch, n_obj) tensor of predicted inclusions. 0=not included, 1=Higgs, 2=W
    # Returns: (n_batch,) tensor of the number of b-tagged sjets in each event
    if use_torch:
        is_sjet = types == 4 # Shape: (n_batch, n_obj)
        THESHOLD = 0.645925
        SMALL_JET_DELTA_R_THRESHOLD = 1.4
        is_btagged = x_part[..., tag_info_idx] > THESHOLD # Shape: (n_batch, n_obj)
        is_btagged_sjet = is_sjet & is_btagged # Shape: (n_batch, n_obj)
        _, Eta_query, Phi_query, _ =  Get_PtEtaPhiM_fromXYZT(x_part[..., 0], x_part[..., 1], x_part[..., 2], x_part[..., 3], use_torch=use_torch) # Each has shape: (n_batch, n_obj)
        _, Eta_key, Phi_key, _ =  Get_PtEtaPhiM_fromXYZT(x_part[..., 0], x_part[..., 1], x_part[..., 2], x_part[..., 3], use_torch=use_torch) #  Each has shape: (n_batch, n_obj)
        delta_R = torch.sqrt((Eta_query.unsqueeze(-2) - Eta_key.unsqueeze(-1))**2 + (Phi_query.unsqueeze(-2) - Phi_key.unsqueeze(-1))**2) # Shape: (n_batch, n_obj, n_obj)
        large_R_W = (pred_inclusion == 2) & (types == 3) # Shape: (n_batch, n_obj)
        large_R_H = (pred_inclusion == 1) & (types == 3) # Shape: (n_batch, n_obj)
        # Fill the delta R where NOT large-R W or large-R Higgs with values greater than the small jet delta R threshold
        delta_R[~einops.repeat((large_R_W | large_R_H), 'batch object_key -> batch object_query object_key', object_query=delta_R.shape[1])] = SMALL_JET_DELTA_R_THRESHOLD + 1 # Shape: (n_batch, n_obj, n_obj)
        passes_delta_R = torch.all(delta_R > SMALL_JET_DELTA_R_THRESHOLD, dim=-1) # Shape: (n_batch, n_obj)
        small_jet_passing = passes_delta_R & is_btagged_sjet # Shape: (n_batch, n_obj)
        return torch.sum(small_jet_passing, dim=-1) # Shape: (n_batch,)
    else:
        is_sjet = types == 4 # Shape: (n_batch, n_obj)
        THESHOLD = 0.645925
        SMALL_JET_DELTA_R_THRESHOLD = 1.4
        is_btagged = x_part[..., tag_info_idx] > THESHOLD # Shape: (n_batch, n_obj)
        is_btagged_sjet = is_sjet & is_btagged # Shape: (n_batch, n_obj)
        _, Eta_query, Phi_query, _ =  Get_PtEtaPhiM_fromXYZT(x_part[..., 0], x_part[..., 1], x_part[..., 2], x_part[..., 3], use_torch=use_torch) # Each has shape: (n_batch, n_obj)
        _, Eta_key, Phi_key, _ =  Get_PtEtaPhiM_fromXYZT(x_part[..., 0], x_part[..., 1], x_part[..., 2], x_part[..., 3], use_torch=use_torch) #  Each has shape: (n_batch, n_obj)
        dEta = np.expand_dims(Eta_query, -2) - np.expand_dims(Eta_key, -1) # Shape: (n_batch, n_obj_Q, n_obj_K)
        dPhi = np.expand_dims(Phi_query, -2) - np.expand_dims(Phi_key,-1) # Shape:  (n_batch, n_obj_Q, n_obj_K)
        dPhi = np.arctan2(np.sin(dPhi), np.cos(dPhi)) # Shape:  (n_batch, n_obj_Q, n_obj_K)
        delta_R = np.sqrt((dEta)**2 + (dPhi)**2) # Shape:  (n_batch, n_obj_Q, n_obj_K)
        large_R_W = (pred_inclusion == 2) & (types == 3) # Shape: (n_batch, n_obj)
        large_R_H = (pred_inclusion == 1) & (types == 3) # Shape: (n_batch, n_obj)
        # Fill the delta R where NOT large-R W or large-R Higgs with values greater than the small jet delta R threshold
        delta_R[~einops.repeat((large_R_W | large_R_H), 'batch object_key -> batch object_query object_key', object_query=delta_R.shape[1])] = SMALL_JET_DELTA_R_THRESHOLD + 1 # Shape: (n_batch, n_obj_Q, n_obj_K)
        passes_delta_R = np.all(delta_R > SMALL_JET_DELTA_R_THRESHOLD, axis=-1) # Shape: (n_batch, n_obj)
        small_jet_passing = passes_delta_R & is_btagged_sjet # Shape: (n_batch, n_obj)
        return np.sum(small_jet_passing, axis=-1) # Shape: (n_batch,)


# %% Temporary testing cell
def GetXYZT_FromPtEtaPhiMOld(pt, eta, phi, m, use_torch=False):
    '''
    Takes in arrays of shape (n_batch[,n_obj],) for Pt, Eta, Phi and M
    of some objects and returns arrays of shape (n_batch[,n_obj],) containing 
    the X, Y, Z, and T(==E) of the objects. Each element of n_batch corresponds
    to one event, and each of the n_objs represents an object in the event.
    '''
    if not use_torch:
        x = pt*np.cos(phi)
        y = pt*np.sin(phi)
        z = pt*np.sinh(eta)
        t = (np.sqrt(x*x + y*y + z*z + m*m))*(m >= 0) + (np.sqrt(np.maximum((x*x + y*y + z*z - m*m), np.zeros(len(m)))))*(m < 0)
        return x, y, z, t
    else:
        x = pt*torch.cos(phi)
        y = pt*torch.sin(phi)
        z = pt*torch.sinh(eta)
        t = (torch.sqrt(x*x + y*y + z*z + m*m))*(m >= 0) + (torch.sqrt(torch.maximum((x*x + y*y + z*z - m*m), torch.zeros(len(m)))))*(m < 0)
        return x, y, z, t

def GetXYZT_FromPtEtaPhiM(pt, eta, phi, m, use_torch=False):
    '''
    Takes in arrays of shape (n_batch[,n_obj],) for Pt, Eta, Phi and M
    of some objects and returns arrays of shape (n_batch[,n_obj],) containing 
    the X, Y, Z, and T(==E) of the objects. Each element of n_batch corresponds
    to one event, and each of the n_objs represents an object in the event.
    '''
    if not use_torch:
        x = pt*np.cos(phi)
        y = pt*np.sin(phi)
        z = pt*np.sinh(eta)
        t = np.sqrt(m**2 + (pt * np.cosh(eta))**2)
        return x, y, z, t
    else:
        x = pt*torch.cos(phi)
        y = pt*torch.sin(phi)
        z = pt*torch.sinh(eta)
        t = torch.sqrt(m**2 + (pt * torch.cosh(eta))**2)
        return x, y, z, t

def GetXYZT_FromPtEtaPhiE(pt, eta, phi, E):
    x,y,z,_=GetXYZT_FromPtEtaPhiM(pt, eta, phi, np.zeros_like(E))
    return x, y, z, E

def Rotate4VectorPhi(x, # (n_batch[,n_obj],)
                  y, # (n_batch[,n_obj],)
                  z, # (n_batch[,n_obj],)
                  t, # (n_batch[,n_obj],)
                  phi, # (n_batch[,n_obj],)
                   ):
    newx = x*np.cos(phi) + y*np.sin(phi)
    newy = -x*np.sin(phi) + y*np.cos(phi)
    return newx, newy, z, t

def Rotate4VectorEta(x, # (n_batch[,n_obj],)
                  y, # (n_batch[,n_obj],)
                  z, # (n_batch[,n_obj],)
                  t, # (n_batch[,n_obj],)
                  eta, # (n_batch[,n_obj],)
                   ):
    theta = 2*np.arctan(np.exp(-eta))
    newx = x*np.cos(theta) + z*np.sin(theta)
    newy = y*np.cos(theta) + z*np.sin(theta)
    newz = -x*np.sin(theta) + y*np.cos(theta)
    return newx, newy, newz, t


def Rotate4VectorPhiEta(x, # (n_batch[,n_obj],)
                  y, # (n_batch[,n_obj],)
                  z, # (n_batch[,n_obj],)
                  t, # (n_batch[,n_obj],)
                  phi, # (n_batch[,n_obj],)
                  eta, # (n_batch[,n_obj],)
                   ):
    # print(x, y, z, t)
    newx, newy, _, _ = Rotate4VectorPhi(x,y,z,t,phi)
    # print(newx, newy, z, t)
    # assert((newy<1e-3).all()) # Only if we're rotating by the angle of the jet itself!
    theta = np.pi/2 - 2*np.arctan(np.exp(-eta))
    # print(theta)
    # print(np.arctan(z/newx))
    # print(np.arctan(newx/z))
    # newx = newx*np.cos(np.pi/2 - theta) + z*np.sin(np.pi/2 - theta) # IT WORKED KINDA WITH THIS
    # newz = -newx*np.sin(np.pi/2 - theta) + z*np.cos(np.pi/2 - theta) # IT WORKED KINDA WITH THIS
    newnewx = newx*np.cos(theta) + z*np.sin(theta)
    newz = -newx*np.sin(theta) + z*np.cos(theta)
    return newnewx, newy, newz, t






# TODO kinda sketch to just have loose stuff if I'm going to import it...
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
dsid_type_pair_to_int = {}
# Also, create dicts to handle different types of decoding:
#   - one to a training label, which will be an integer to be one-hot encoded
#   - the other to an evaluation label, which will tell us more info about the sample for plotting/testing performance
decode_int_to_training_label = {}
decode_int_to_evaluation_label = {}
counter = 0  # Start integer for mapping
for x in dsid_set:
    for y in types_set:
        dsid_type_pair_to_int[(x, y)] = counter
        if (500000<x) and (x<600000):
            if y == 1:
                decode_int_to_training_label[counter] = 1
            elif (y==-2) or (y==2):
                decode_int_to_training_label[counter] = 2
            else:
                assert(False)
        else:
            decode_int_to_training_label[counter] = 0
        decode_int_to_evaluation_label[counter] = [x,y] # Just put all info in for now
        counter += 1

dsid_type_int_to_pair = {v: k for k, v in dsid_type_pair_to_int.items()} # Create inverse dictionary
mapping_array_pair_to_int = np.zeros((len(dsid_set), len(types_set)), dtype=int)
for (x, y), val in dsid_type_pair_to_int.items():
    i = np.where(dsid_set == x)[0][0]
    j = np.where(types_set == y)[0][0]
    mapping_array_pair_to_int[i, j] = val

def decode_y_eval_to_info(y_int_array):
    # Extract DSID and decay_mode for each entry in y_int_array
    dsid_array = np.array([decode_int_to_evaluation_label[y][0] for y in y_int_array])
    decay_mode_array = np.array([decode_int_to_evaluation_label[y][1] for y in y_int_array])
    # Initialize mass and decay_mode_real arrays with default values
    mass_array = np.zeros_like(dsid_array, dtype=float)
    decay_mode_real_array = np.zeros_like(decay_mode_array, dtype=int)
    # Apply conditions for DSID range (500000 < dsid < 600000)
    valid_dsid_mask = (500000 < dsid_array) & (dsid_array < 600000)
    # For valid DSID values, lookup mass and calculate decay_mode_real
    mass_array[valid_dsid_mask] = np.vectorize(DSID_MASS_MAPPING.get)(dsid_array[valid_dsid_mask])
    # Define decay_mode_real based on decay_mode values
    decay_mode_real_array[valid_dsid_mask & (decay_mode_array == 1)] = 1
    decay_mode_real_array[valid_dsid_mask & ((decay_mode_array == 2) | (decay_mode_array == -2))] = 2
    return mass_array, decay_mode_real_array, dsid_array