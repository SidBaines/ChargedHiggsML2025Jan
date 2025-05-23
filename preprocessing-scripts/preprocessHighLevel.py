# %% Add path to local files in case we're running in a different directory
if 0: # Don't think I need this anymore
    import sys
    sys.path.insert(0,'/users/baines/Code/ChargedHiggs_ExperimentalML/')
# %% [markdown]
# # Load required modules
import time
ts = []
ts.append(time.time())

import numpy as np
from utils.utils import read_file
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from jaxtyping import Float, Int
import einops
import matplotlib.pyplot as plt
# from utils.utils import Get_PtEtaPhiM_fromXYZT, GetXYZT_FromPtEtaPhiM, GetXYZT_FromPtEtaPhiE, Rotate4VectorPhi, Rotate4VectorEta, Rotate4VectorPhiEta
# import datasets

# %% Some basic setup
# Some choices about the  process
TOSS_UNCERTAIN_TRUTH = True
if not TOSS_UNCERTAIN_TRUTH:
    raise NotImplementedError # Need to work out what to do (eg. put in a flag so they're not used as training?)
USE_OLD_TRUTH_SETTING = True
# if USE_OLD_TRUTH_SETTING:
#     raise NotImplementedError # Need to check if we should require truth_agreement variable here or not
MET_CUT_ON = True
MH_SEL = False
REQUIRE_XBB = True
BIN_WRITE_TYPE=np.float32
OUTPUT_DIR = '/data/atlas/baines/20250322v4_highLevel' + '_MetCut'*MET_CUT_ON + '_mHSel'*MH_SEL + '_OldTruth'*USE_OLD_TRUTH_SETTING + '_RemovedUncertainTruth'*TOSS_UNCERTAIN_TRUTH +  '/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
types_set = np.array([-2, 1, 2])  # Try not to change this often - have to re-binarise if we do!

# Create a mapping from the dsid/decay-type pair to integers, for the purposes of binarising data.
dsid_set = np.array([363355,363356,363357,363358,363359,363360,363489,407342,407343,407344,
            407348,407349,410470,410646,410647,410654,410655,411073,411074,411075,
            411077,411078,412043,413008,413023,510115,510116,510117,510118,510119,
            510120,510121,510122,510123,510124,700320,700321,700322,700323,700324,
            700325,700326,700327,700328,700329,700330,700331,700332,700333,700334,
            700338,700339,700340,700341,700342,700343,700344,700345,700346,700347,
            700348,700349,
            ]) # Try not to change this often - have to re-binarise if we do!


# Running statistics (mean, variance, count) for each input feature
class RunningStats:
    def __init__(self, n_features):
        self.n = 0  # Count of data points seen
        self.mean = np.zeros(n_features)  # Mean for each feature
        self.m2 = np.zeros(n_features)  # Sum of squared differences for variance
    
    def update(self, x):
        # x is the batch of data, where each row is a data point, and columns are features
        n_batch = x.shape[0]
        
        # Update count
        self.n += n_batch
        if n_batch > 0:    
            # Update mean
            delta = x - self.mean
            self.mean += delta.sum(axis=0) / self.n
            
            # Update m2 (sum of squared differences)
            delta2 = x - self.mean
            self.m2 += (delta * delta2).sum(axis=0)
    
    def compute_std(self):
        # Calculate standard deviation using the variance (m2 / n - 1)
        return np.sqrt(self.m2 / (self.n - 1))

    def get_mean(self):
        return self.mean
    
    def get_variance(self):
        return self.m2 / (self.n - 1)  # Variance for each feature

# Initialize running stats for 7 features (based on your input features)
n_features = 7  # Adjust to match the number of input features you're tracking
running_stats = {
    'lvbb':RunningStats(n_features),
    'qqbb':RunningStats(n_features),
}

# %%
# Main function to actually process a single file and return the arrays
def process_single_file(filepath, target_channel):
    assert((target_channel == 'lvbb') or (target_channel == 'qqbb'))
    if USE_OLD_TRUTH_SETTING:
        truth_var = 'truth_W_decay_mode'
        mH_var = 'mH'
    else:
        truth_var = 'll_truth_decay_mode'
        mH_var = 'll_best_mH'
    event_level_features=['eventWeight',
        # 'selection_category',
        'combined_category',
        'MET',
        mH_var]
    if target_channel == 'lvbb':
        event_level_features += [
            'mVH_lvbb',
            # lvbb neural net inputs
            'deltaR_LH','deltaPhi_HWlep','deltaEta_HWlep','ratio_Hpt_mVH_lvbb','ratio_Wpt_mVH_lvbb','mTop_lepto','LepEnergyFracLep',
        ]
    elif target_channel == 'qqbb':
        event_level_features += [
            'mVH_qqbb',
            #qqbb neural net inputs
            'deltaR_LH','deltaPhi_HWhad','deltaEta_HWhad','ratio_Hpt_mVH_qqbb','ratio_Wpt_mVH_qqbb','deltaR_LWhad','LepEnergyFracHad',
            ]
    from utils.utils import read_file
    _, x_event, y = read_file(filepath, 
                                particle_features=[],
                                event_level_features=event_level_features,
                                labels=['DSID', truth_var, 'eventNumber'],
                                new_inputs_labels=True
    )
    if len(y):
        dsid = y[0,0]
    else:
        dsid = 0 # Let the function run with empty arrays anyway
    eventNumber = y[:,2]
    if 0:
        selection_category = x_event[:,1]
        if target_channel == 'lvbb':
            channel_sel = ((selection_category == 0) | (selection_category == 8) | (selection_category == 10))
            truth_sel = (y[:,1]==1) if ((dsid>500000) and (dsid<600000)) else np.ones_like(y[:,1]).astype(bool)
        elif target_channel == 'qqbb':
            channel_sel = ((selection_category == 3) | (selection_category == 9))
            truth_sel = (y[:,1]==2) if ((dsid>500000) and (dsid<600000)) else np.ones_like(y[:,1]).astype(bool)
    else:
        if target_channel == 'lvbb':
            channel_sel = (x_event[:,1] == 0)
            truth_sel = (y[:,1]==1) if ((dsid>500000) and (dsid<600000)) else np.ones_like(y[:,1]).astype(bool)
        elif target_channel =='qqbb':
            channel_sel = (x_event[:,1] == 3)
            truth_sel = (y[:,1]==2) if ((dsid>500000) and (dsid<600000)) else np.ones_like(y[:,1]).astype(bool)
    is_sig = ((y[:,0] > 500000) & (y[:,0] < 600000))
    if MET_CUT_ON:
        met_sel = x_event[:,2] > 30e3
    else:
        met_sel = np.ones_like(x_event[:,2]).astype(bool)
    if MH_SEL:
        mH_sel = (x_event[:,3] >= 95) & (x_event[:,3] <= 140)
    else:
        mH_sel = np.ones_like(x_event[:,3]).astype(bool)
    if TOSS_UNCERTAIN_TRUTH:
        # uncertain_sel = ~((np.searchsorted(types_set, y[:, 1])==0) & is_sig)
        uncertain_sel = ~(((y[:, 1] != 1) & (y[:, 1] != 2)) & is_sig)
    else:
        uncertain_sel = np.ones_like(is_sig)#.astype(bool)
    combined_sel = channel_sel & met_sel & truth_sel & mH_sel & uncertain_sel
    removals = np.bincount(is_sig[~(combined_sel)])
    wts = x_event[combined_sel,0]
    mH = x_event[combined_sel,3]
    if USE_OLD_TRUTH_SETTING:
        mH = mH*1e3
    mWH = x_event[combined_sel,4]
    x = x_event[combined_sel,5:]
    dsids = y[combined_sel,0]
    eventNumber = eventNumber[combined_sel]
    # truth_labels = np.searchsorted(types_set, y[combined_sel, 1])
    truth_labels = (y[combined_sel, 1] == 1)*1 + (y[combined_sel, 1] == 2)*2

    return x, truth_labels, wts, dsids, mWH, mH, removals, eventNumber

def combine_arrays_for_writing(x_chunk, y_chunk, weights_chunk, mWh_chunk, dsids_chunk, mHs_chunk, eventNumber_chunk):
    array_to_write=np.float32(np.concatenate(
        [y_chunk.reshape(-1,1), 
         weights_chunk.reshape(-1,1), 
         mWh_chunk.reshape(-1,1), 
         dsids_chunk.reshape(-1,1), 
         mHs_chunk.reshape(-1,1), 
         (eventNumber_chunk%1000).reshape(-1,1), # Have to make this remainder some small-ish number, otherwise the numbers might be too big for the np.float32 and will come out as zero. Currently plan to just use for train/val split, so definitely shouldn't need more than eg. 1000
         x_chunk,
         ],
    axis=-1
    ))
    np.random.shuffle(array_to_write)
    return array_to_write


# %%
types_dict = {0: 'electron', 1: 'muon', 2: 'neutrino', 3: 'ljet', 4: 'sjet', 5: 'Xbb_ljet'}
# DATA_PATH='/data/atlas/HplusWh/20241218_SeparateLargeRJets_NominalWeights/'
DATA_PATH='/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/'
DATA_PATH='/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/'
DATA_PATH='/data/atlas/HplusWh/20250313_WithTrueInclusion_FixedOverlapWHsjet_SmallJetCloseToLargeJetRemovalDeltaR0.5/'
MAX_CHUNK_SIZE = 100000
# MAX_PER_DSID = {dsid : 10000000 for dsid in dsid_set}
# MAX_PER_DSID[410470] = 100

for dsid in dsid_set:
    for channel in ['lvbb', 'qqbb']:
        # # if dsid >= 400000:
        # if dsid != 510115:
        #     continue
        if True:
            pass
        else:
            continue
        removals = {0:0, 1:0, 2:0}
        nfs = 0
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
        mWHs = []
        mHs = []
        dsids = []
        eventNumbers = []
        current_chunk_size = 0
        total_events_written_for_sample = 0
        total_entries_written_for_sample = 0
        sum_abs_weights_written_for_sample = 0
        sum_weights_written_for_sample = 0
        memmap_path = os.path.join(OUTPUT_DIR, f'dsid_{dsid}_{channel}.memmap')
        if len(all_files) > 0: # Safeguard for when there aren't any files to loop through, so we don't create an empty memmap file
            with open(memmap_path, 'wb') as f:
                pass  # Create empty file
        for file_n, path in enumerate(all_files):
            # print(path)
            # if path == '/data/atlas/HplusWh/20241128_ProcessedLightNtuples/user.rhulsken.mc16_13TeV.363355.She221_ZqqZvv.TOPQ1.e5525s3126r10201p4512.Nominal_v0_1l_out_root/user.rhulsken.31944615._000001.out.root':
            #     continue
            x_chunk, y_chunk, weights_chunk, dsid_chunk, mWhs_chunk, mH_chunk, removals_chunk, eventNumber_chunk = process_single_file(filepath=path, target_channel=channel)
            means=np.load(f'/data/atlas/baines/tmp2_highLevel_MetCut/{channel}_mean.npy')
            stds=np.load(f'/data/atlas/baines/tmp2_highLevel_MetCut/{channel}_std.npy')
            # if x_chunk.shape[0]>500:
            #     print(x_chunk.shape)
            #     print(((x_chunk - means)/stds).mean(axis=0))
            #     print(((x_chunk - means)/stds).std(axis=0))
            running_stats[channel].update(x_chunk)
            
            x_parts.append(x_chunk)
            ys.append(y_chunk)
            weights.append(weights_chunk)
            mWHs.append(mWhs_chunk)
            mHs.append(mH_chunk)
            dsids.append(dsid_chunk)
            eventNumbers.append(eventNumber_chunk)
            current_chunk_size += x_chunk.shape[0]
            if (current_chunk_size > MAX_CHUNK_SIZE) or (file_n+1 == len(all_files)):
                array_chunk = combine_arrays_for_writing(x_chunk=np.concatenate(x_parts, axis=0), y_chunk=np.concatenate(ys, axis=0), weights_chunk=np.concatenate(weights, axis=0), mWh_chunk=np.concatenate(mWHs, axis=0), dsids_chunk=np.concatenate(dsids, axis=0), mHs_chunk=np.concatenate(mHs, axis=0), eventNumber_chunk=np.concatenate(eventNumbers, axis=0))
                # assert(False)
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
                mWHs = []
                mHs = []
                dsids = []
                eventNumbers = []
            # assert(False)
            for i in range(len(removals_chunk)):
                removals[i]+=removals_chunk[i]
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
        print("Total accepted %s: %d" %(channel, total_events_written_for_sample))
        print("Total bkg removed for no ljet/MET Cut/Selection category fail: %d" %(removals[0]))
        print("Total lvbb removed for no ljet/MET Cut/Selection category fail: %d" %(removals[1]))
        print("Total qqbb removed for no ljet/MET Cut/Selection category fail: %d" %(removals[2]))

for channel in ['lvbb', 'qqbb']:
    print("Feature means: ", running_stats[channel].get_mean())
    print("Feature std devs: ", running_stats[channel].compute_std())
    # Optionally, you can save these stats for later use in a scaler
    np.save(os.path.join(OUTPUT_DIR, f'{channel}_mean.npy'), running_stats[channel].get_mean())
    np.save(os.path.join(OUTPUT_DIR, f'{channel}_std.npy'), running_stats[channel].compute_std())



# %%
