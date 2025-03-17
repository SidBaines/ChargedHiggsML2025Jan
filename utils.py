import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from typing import List
import matplotlib as mpl

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

def basic_lr_scheduler(epoch: int, lr_high: float, lr_low: float, n_epochs: int, log=True):
    """
    This function calculates the learning rate following a flat decreasing schedule
    """
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

# %% Temporary testing cell
def GetXYZT_FromPtEtaPhiM(pt, eta, phi, m):
    '''
    Takes in arrays of shape (n_batch[,n_obj],) for Pt, Eta, Phi and M
    of some objects and returns arrays of shape (n_batch[,n_obj],) containing 
    the X, Y, Z, and T(==E) of the objects. Each element of n_batch corresponds
    to one event, and each of the n_objs represents an object in the event.
    '''
    x = pt*np.cos(phi)
    y = pt*np.sin(phi)
    z = pt*np.sinh(eta)
    t = (np.sqrt(x*x + y*y + z*z + m*m))*(m >= 0) + (np.sqrt(np.maximum((x*x + y*y + z*z - m*m), np.zeros(len(m)))))*(m < 0)
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