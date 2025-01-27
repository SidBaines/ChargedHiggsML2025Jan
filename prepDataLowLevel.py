# %% Add path to local files in case we're running in a different directory
import sys
sys.path.insert(0,'/users/baines/Code/ChargedHiggs_ProcessingForIntNote/')
# %% [markdown]
# # Load required modules
import time
ts = []
ts.append(time.time())

import numpy as np
from mydataloader import read_file
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from jaxtyping import Float, Int
import einops
import matplotlib.pyplot as plt
from utils import Get_PtEtaPhiM_fromXYZT, GetXYZT_FromPtEtaPhiM, GetXYZT_FromPtEtaPhiE, Rotate4VectorPhi, Rotate4VectorEta, Rotate4VectorPhiEta
# import datasets

# %% Some basic setup
# Some choices about the  process
SHUFFLE_OBJECTS = False
CONVERT_TO_PT_PHI_ETA_M = False
MET_CUT_ON = True
REQUIRE_XBB = True
N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
N_CTX = 7 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root files we're reading in.
BIN_WRITE_TYPE=np.float32
max_n_objs = 14 # BE CAREFUL because this might change and if it does you ahve to rebinarise
OUTPUT_DIR = '/data/atlas/baines/tmp_SingleXbbSelected_XbbTagged_WithRecoMasses_' + 'semi_shuffled_'*SHUFFLE_OBJECTS + f'{max_n_objs}' + '_PtPhiEtaM'*CONVERT_TO_PT_PHI_ETA_M + '_MetCut'*MET_CUT_ON + '_XbbRequired'*REQUIRE_XBB + '/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_Real_Vars=4 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise

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
    return mass_array, decay_mode_real_array






# %%
# Main function to actually process a single file and return the arrays
def process_single_file(filepath, max_n_objs, shuffle_objs):
    x_part, x_event, y = read_file(filepath, 
                                particle_features=[
                                    #  'part_pt', 'part_eta', 'part_phi', 'part_energy', 'type',
                                    'part_px', 'part_py', 'part_pz', 'part_energy', 'll_particle_type',
                                                    # 'part_isChargedHadron',
                                                    # 'part_isNeutralHadron',
                                                    # 'part_isPhoton',
                                                    # 'part_isElectron',
                                                    # 'part_isMuon',
                                                    ],
                                event_level_features=[
                                    'eventWeight',
                                    'selection_category',
                                    'MET',
                                    'll_best_mWH_qqbb',
                                    'll_best_mWH_lvbb',

                                ],
                                labels=['DSID', 'll_truth_decay_mode'],
                                new_inputs_labels=True
    )
    type_part = x_part[:,N_Real_Vars,:]
    type_part[(x_part[:,:N_Real_Vars,:]==0).all(axis=1)] = -1
    # Types dict: 0: electron, 1: muon, 2: neutrino, 3: ljet, 4: sjet, 5: XbbTaggedLjet
    lepton_locs = np.where(((type_part==0)|(type_part==1)))[1]
    lep_XYZT = x_part[np.arange(x_part.shape[0]), :4, lepton_locs]
    lep_pt, lep_eta, lep_phi, _ = Get_PtEtaPhiM_fromXYZT(lep_XYZT[:,0], lep_XYZT[:,1], lep_XYZT[:,2], lep_XYZT[:,3])
    # if MET_CUT_ON:
    #     neutrino_locs = np.where(type_part==2)[1]
    #     neutrino_XYZT = x_part[np.arange(x_part.shape[0]), :4, neutrino_locs]
    #     Neutrino_Pt, _, _, _ = Get_PtEtaPhiM_fromXYZT(neutrino_XYZT[:,0], neutrino_XYZT[:,1], neutrino_XYZT[:,2], neutrino_XYZT[:,3])
    lep_phi_expanded = einops.repeat(lep_phi,'b -> b o',o=x_part.shape[-1])
    lep_eta_expanded = einops.repeat(lep_eta,'b -> b o',o=x_part.shape[-1])
    if 0:
        rotatedx, rotatedy, rotatedz, _ = Rotate4VectorPhiEta(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded, 
                        lep_eta_expanded
                        )
        # rotated_lep_x, _, _, _ = Rotate4VectorPhiEta(*GetXYZT_FromPtEtaPhiE(x_event[:,0],x_event[:,1],x_event[:,2],x_event[:,3]), x_event[:,2], x_event[:,1])
        x_part[:,0,:] = rotatedx - einops.repeat(rotated_lep_x, 'b->b o',o=x_part.shape[-1])
        x_part[:,1,:] = rotatedy
        x_part[:,2,:] = rotatedz
        x_part[:,3,:] = x_part[:,3,:]/ einops.repeat(lep_e, 'b->b o',o=x_part.shape[-1])
    elif 0:
        rotatedx, rotatedy, rotatedz, _ = Rotate4VectorPhiEta(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded, 
                        lep_eta_expanded
                        )
        # rotated_lep_x, _, _, _ = Rotate4VectorPhiEta(*GetXYZT_FromPtEtaPhiE(x_event[:,0],x_event[:,1],x_event[:,2],x_event[:,3]), x_event[:,2], x_event[:,1])
        x_part[:,0,:] = rotatedx - einops.repeat(rotated_lep_x, 'b->b o',o=x_part.shape[-1])
        x_part[:,1,:] = rotatedy
        x_part[:,2,:] = rotatedz
    elif 0:
        rotatedx, rotatedy, rotatedz, _ = Rotate4VectorPhiEta(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded, 
                        lep_eta_expanded
                        )
        x_part[:,0,:] = rotatedx
        x_part[:,1,:] = rotatedy
        x_part[:,2,:] = rotatedz
    elif 1:
        rotatedx, rotatedy, _, _ = Rotate4VectorPhi(
                        x_part[:,0,:], 
                        x_part[:,1,:], 
                        x_part[:,2,:], 
                        x_part[:,3,:], 
                        lep_phi_expanded
                        )
        x_part[:,0,:] = rotatedx
        x_part[:,1,:] = rotatedy
    if CONVERT_TO_PT_PHI_ETA_M:
        pt, eta, phi, m = Get_PtEtaPhiM_fromXYZT(x_part[:,0,:],x_part[:,1,:],x_part[:,2,:],x_part[:,3,:])
        x_part[:,0,:] = pt
        x_part[:,2,:] = eta
        x_part[:,1,:] = phi
        x_part[:,3,:] = m
    type_part[type_part==-1] = N_CTX-1 # Set the non-existing particles to be last (dummy) index of embedding tensor
    x_part = np.concatenate(
        [
            einops.rearrange(type_part,'b o -> b 1 o'),
            x_part[:,:N_Real_Vars,:]
        ],
        axis=1
    )

    event_weights = x_event[:,0]
    selection_category = x_event[:,1]
    mWh_qqbb = x_event[:,3]
    mWh_lvbb = x_event[:,4]
    if MET_CUT_ON:
        met = x_event[:,2]

    # Remove events with no large-R jets and which don't pass selection category and which have too low Met Pt
    selection_removals = ~((selection_category == 0) | (selection_category == 3) | (selection_category == 8) | (selection_category == 9) | (selection_category == 10))
    is_sig = ((y[:,0] > 500000) & (y[:,0] < 600000))
    if not REQUIRE_XBB:
        no_ljet = (((type_part==3)|(type_part==5)).sum(axis=1) == 0)
    else:
        no_ljet = ((type_part==5).sum(axis=1) == 0)
    if MET_CUT_ON:
        low_MET = met < 30e3 # 30GeV Minimum for MET
    else:
        low_MET = np.ones_like(no_ljet).astype(bool)
    removals = {0: ((no_ljet | selection_removals | low_MET) & (~is_sig)).sum(),
                1: ((no_ljet | selection_removals | low_MET) & is_sig).sum()}
    x_part = x_part[~(no_ljet | selection_removals | low_MET)]
    y = y[~(no_ljet | selection_removals | low_MET)]
    event_weights = event_weights[~(no_ljet | selection_removals | low_MET)]
    mWh_qqbb = mWh_qqbb[~(no_ljet | selection_removals | low_MET)]
    mWh_lvbb = mWh_lvbb[~(no_ljet | selection_removals | low_MET)]
    type_part = type_part[~(no_ljet | selection_removals | low_MET)]
    

    # HERE Need to write some code to store this properly
    first_indices = np.searchsorted(dsid_set, y[:, 0])
    second_indices = np.searchsorted(types_set, y[:, 1])
    y_mapped = mapping_array_pair_to_int[first_indices, second_indices]

    # Reshape the x array to how we want to read it in later
    x_part = einops.rearrange(x_part, 'batch d_input object -> batch object d_input')
    if shuffle_objs: # # Shuffle the tensors along the objects dimension (keeping the mapping bettween X_train objects and types_train objects the same)
        # non_zero_mask = x[:, :, 0] != 0  # Mask for non-zero elements in the first variable
        # permutation_indices = torch.argsort(torch.rand(*x_part[:,:,0].shape), dim=-1)
        # batch_indices = torch.arange(x_part.shape[0]).unsqueeze(1).expand(x_part.shape[0], x_part.shape[1])
        # x_part = x_part[batch_indices, permutation_indices]
        num_non_zero = (x_part[:,:,0]!=N_CTX-1).sum(axis=1)
        result = x_part.copy()
        if 1: # Old style - shuffle only the non-zero ones
            for i in range(x_part.shape[0]):  # For each batch element
                n_non_zero = num_non_zero[i]
                if n_non_zero > 1:
                    # Generate permutation indices for the non-zero objects in this batch
                    permuted_indices = np.random.permutation(n_non_zero)
                    # Shuffle the non-zero elements along the object dimension
                    result[i,:n_non_zero] = result[i, permuted_indices]
        else: # Shuffle all up to max_n objs
            for i in range(x_part.shape[0]):  # For each batch element
                n_non_zero = num_non_zero[i]
                if n_non_zero > 1:
                    # Generate permutation indices for the non-zero objects in this batch
                    permuted_indices = np.random.permutation(max_n_objs)
                    # Shuffle the non-zero elements along the object dimension
                    result[i,:max_n_objs] = result[i, permuted_indices]
        return result[:,:max_n_objs,:N_Real_Vars+1], y_mapped, event_weights, removals, mWh_qqbb, mWh_lvbb
    else:
        return x_part[:,:max_n_objs,:N_Real_Vars+1], y_mapped, event_weights, removals, mWh_qqbb, mWh_lvbb

def combine_arrays_for_writing(x_chunk, y_chunk, weights_chunk, mWh_qqbbs_chunk, mWh_lvbbs_chunk):
    # print(type(y_chunk))
    # print(y_chunk.shape)
    y_chunk = einops.repeat(y_chunk,'b -> b 1 nvars', nvars=x_chunk.shape[-1]).astype(BIN_WRITE_TYPE)
    # print(type(y_chunk))
    # print(y_chunk.shape)
    y_chunk[:, 0, 1] = weights_chunk.squeeze()
    y_chunk[:, 0, 2] = mWh_qqbbs_chunk.squeeze()
    y_chunk[:, 0, 3] = mWh_lvbbs_chunk.squeeze()
    array_to_write=np.float32(np.concatenate(
        [
            y_chunk,
            x_chunk
        ],
    axis=-2
    ))
    np.random.shuffle(array_to_write)
    return array_to_write


# %%
types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'Xbb_ljet'}
# DATA_PATH='/data/atlas/HplusWh/20241218_SeparateLargeRJets_NominalWeights/'
DATA_PATH='/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/'
MAX_CHUNK_SIZE = 100000
# MAX_PER_DSID = {dsid : 10000000 for dsid in dsid_set}
# MAX_PER_DSID[410470] = 100
for dsid in dsid_set:
    # if dsid != 363489:
    if dsid >= 700340:
        continue
# for dsid in [510120]:
    # if ((500000<dsid) and (600000>dsid)) or (dsid==410470):
    # if (dsid==410470):
    # if (dsid==700349):
    if True:
    # if ((500000<dsid) and (600000>dsid)):
    # if (dsid==410470):
    # # if (dsid==510120):
    # if dsid > 700338:
    # if (dsid ==510121):
    # if ((dsid < 600000) and (dsid > 500000)) or (dsid==410470) or (dsid==407342):
        pass
    else:
        continue
    nfs = 0
    removals = {0:0, 1:0}
    # if ((dsid > 500000) and (dsid < 600000)):# or (dsid==410470):
    # if (dsid==410470):
    #     pass
    # else:
    #     continue
    all_files = []
    for filename in os.listdir(DATA_PATH):
         if (str(dsid) in filename) and (filename.endswith('.root')):
            all_files.append(DATA_PATH + '/' + filename)
    # if dsid != 510122:
    #     continue
    x_parts=[]
    ys=[]
    weights = []
    mWH_qqbbs = []
    mWH_lvbbs = []
    current_chunk_size = 0
    total_events_written_for_sample = 0
    total_entries_written_for_sample = 0
    sum_abs_weights_written_for_sample = 0
    sum_weights_written_for_sample = 0
    memmap_path = os.path.join(OUTPUT_DIR, f'dsid_{dsid}.memmap')
    if len(all_files) > 0: # Safeguard for when there aren't any files to loop through, so we don't create an empty memmap file
        with open(memmap_path, 'wb') as f:
            pass  # Create empty file
    for file_n, path in enumerate(all_files):
        # print(path)
        # if path == '/data/atlas/HplusWh/20241128_ProcessedLightNtuples/user.rhulsken.mc16_13TeV.363355.She221_ZqqZvv.TOPQ1.e5525s3126r10201p4512.Nominal_v0_1l_out_root/user.rhulsken.31944615._000001.out.root':
        #     continue
        x_chunk, y_chunk, weights_chunk, removals_chunk, mWh_qqbb_chunk, mWh_lvbb_chunk = process_single_file(filepath=path, max_n_objs=max_n_objs, shuffle_objs=SHUFFLE_OBJECTS)
        x_parts.append(x_chunk)
        ys.append(y_chunk)
        weights.append(weights_chunk)
        mWH_qqbbs.append(mWh_qqbb_chunk)
        mWH_lvbbs.append(mWh_lvbb_chunk)
        current_chunk_size += x_chunk.shape[0]
        if (current_chunk_size > MAX_CHUNK_SIZE) or (file_n+1 == len(all_files)):
            array_chunk = combine_arrays_for_writing(x_chunk=np.concatenate(x_parts, axis=0), y_chunk=np.concatenate(ys, axis=0), weights_chunk=np.concatenate(weights, axis=0), mWh_qqbbs_chunk=np.concatenate(mWH_qqbbs, axis=0), mWh_lvbbs_chunk=np.concatenate(mWH_lvbbs, axis=0))
            # assert(False)
            if array_chunk.shape[0] > 0: # Check there's actually something in there
                # memmap = np.memmap(memmap_path, dtype=BIN_WRITE_TYPE, mode='r+', offset=total_entries_written_for_sample*array_chunk.itemsize, shape=array_chunk.shape)
                memmap = np.memmap(memmap_path, dtype=BIN_WRITE_TYPE, mode='r+', offset=total_entries_written_for_sample*array_chunk.itemsize, shape=array_chunk.shape)
                memmap[:] = array_chunk[:]
            total_events_written_for_sample += array_chunk.shape[0]
            total_entries_written_for_sample += np.prod(array_chunk.shape)
            sum_abs_weights_written_for_sample += np.abs(np.concatenate(weights, axis=0)).sum()
            sum_weights_written_for_sample += np.concatenate(weights, axis=0).sum()
            # Save total number of events
            with open(memmap_path + '.shape', 'w') as f:
                f.write(f"{total_events_written_for_sample},{sum_abs_weights_written_for_sample},{sum_weights_written_for_sample}")
            current_chunk_size = 0
            x_parts=[]
            ys=[]
            weights = []
            mWH_qqbbs = []
            mWH_lvbbs = []
        removals[0]+=removals_chunk[0]
        removals[1]+=removals_chunk[1]
        nfs+=1
        if (nfs%10)==0:
            print("Processed %d files, have %d events in buffer, %d written so far" %(nfs, current_chunk_size, total_events_written_for_sample))
        if nfs > 1000000:
            break
        # if (total_events_written_for_sample > MAX_PER_DSID[dsid]):
        #     print("Reached %d for %s" %(total_events_written_for_sample, filename))
        #     break
    with open(memmap_path + '.shape', 'w') as f:
        f.write(f"{total_events_written_for_sample},{sum_abs_weights_written_for_sample},{sum_weights_written_for_sample}")
    print("Total accepted: %d" %(total_events_written_for_sample))
    print("Total bkg removed for no ljet/MET Cut/Selection category fail: %d" %(removals[0]))
    print("Total sig removed for no ljet/MET Cut/Selection category fail: %d" %(removals[1]))



# %%
