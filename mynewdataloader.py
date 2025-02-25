# %%
import numpy as np
import os
import random
from typing import List, Dict, Tuple
import torch
from math import ceil
import einops
import typing
from math import floor
from sklearn.utils import shuffle
from copy import copy


dsid_set = np.array([363355,363356,363357,363358,363359,363360,363489,407342,407343,407344,
            407348,407349,410470,410646,410647,410654,410655,411073,411074,411075,
            411077,411078,412043,413008,413023,510115,510116,510117,510118,510119,
            510120,510121,510122,510123,510124,700320,700321,700322,700323,700324,
            700325,700326,700327,700328,700329,700330,700331,700332,700333,700334,
            700338,700339,700340,700341,700342,700343,700344,700345,700346,700347,
            700348,700349,
            ]) # Try not to change this often - have to re-binarise if we do!
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

class ProportionalMemoryMappedDataset:
    def __init__(self, 
                 memmap_paths: Dict[int, str],  # DSID to memmap path
                 max_objs_in_memmap: int = -1,
                 N_Real_Vars: int = 4,
                 class_proportions: Dict[int, float] = None,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 is_train: bool = True,
                 train_split: float = 0.8,
                 n_targets: int = 3,
                 shuffle: bool = False, # Whether or not to shuffle along the object dimension,
                 shuffle_batch: bool = True, # Whether or not to shuffle along the batch dimension,
                 signal_reweights: typing.Optional[np.array]=None,
                 means=None,
                 stds=None,
                 objs_to_output=14,
                 ):
        """
        Initialize a dataset loader with memory-mapped files for each class (DSID)
        
        Args:
        - memmap_paths: Dictionary mapping DSID to path of memory-mapped file
        - class_proportions: Optional dictionary of sampling proportions for each DSID
        - batch_size: Number of samples per batch
        """
        # Load memory-mapped files
        assert(max_objs_in_memmap > 0)
        self.memmaps = {}
        self.sample_counts = {}
        self.abs_weight_sums = {}
        self.weight_sums = {}
        self.device = device
        self.n_targets = n_targets
        self.shuffle_objects = shuffle
        self.shuffle_batch = shuffle_batch
        self.signal_reweights = signal_reweights
        self.objs_to_output = objs_to_output

        # Add train/val splitting logic
        self.is_train = is_train
        self.train_split = train_split
        # Add/populate with defaults the mean/std for scaling inputs
        if means is not None:
            self.means = torch.Tensor(means).to(torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            self.means = torch.zeros(N_Real_Vars).to(torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
        if stds is not None:
            self.stds = torch.Tensor(stds).to(torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            self.stds = torch.ones(N_Real_Vars).to(torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)

        self.bkg_weight_sums = 0
        # self.sig_weight_sums = 0
        self.num_signals = 0
        
        for dsid, path in memmap_paths.items():
            # Assume each memmap has a companion .shape file with total number of samples
            shape_path = path + '.shape'
            with open(shape_path, 'r') as f:
                metadata = f.read()
            metadata.split(',')
            total_samples, sum_abs_weights, sum_weights = metadata.split(',')
            total_samples=int(total_samples)
            sum_abs_weights=float(sum_abs_weights)
            sum_weights=float(sum_weights)
            
            # Memory map the file
            # print(total_samples)
            # print(path)
            # print(path)
            if total_samples == 0:
                print(f"Skipping dsid {dsid} for too few ({total_samples}) samples")
            else:
                self.sample_counts[dsid] = total_samples
                self.memmaps[dsid] = np.memmap(
                    path, 
                    dtype=np.float32,  # adjust dtype as needed
                    mode='r+', 
                    shape=(total_samples, max_objs_in_memmap+2, N_Real_Vars+1)  # adjust shape as per your data
                )
                self.abs_weight_sums[dsid] = sum_abs_weights
                self.weight_sums[dsid] = sum_weights
                if (dsid<500000) or (dsid > 600000):
                    self.bkg_weight_sums += sum_weights
                else:
                    # self.sig_weight_sums += sum_weights
                    self.num_signals += 1
        if self.signal_reweights is not None:
            assert(len(self.signal_reweights)==self.num_signals)
        
        # Determine proportions
        if class_proportions is None:
            total_samples = sum(self.sample_counts.values())
            class_proportions = {
                dsid: count/total_samples 
                for dsid, count in self.sample_counts.items()
            }
        self.class_proportions = class_proportions
        
        # samples_per_class = {
        #     dsid: max(1, ceil(self.batch_size * prop)) 
        #     for dsid, prop in self.class_proportions.items()
        # }
        # for dsid in samples_per_class.keys():
        #     # if samples_per_class[dsid] < len(self):
        #     if samples_per_class[dsid]
        #         print("REMOVING %d for too few samples" %(dsid))
        
        self.batch_size = batch_size
        
        # Precompute sample indices for each class
        self._reset_indices()
    
    def _reset_indices(self):
        """
        Reset and initialize indices for sampling
        """
        self.current_indices = {}
        for dsid in self.memmaps.keys():
            # Randomly shuffle indices for each class
            if self.is_train:
                self.current_indices[dsid] = list(range(floor((1-self.train_split)*self.sample_counts[dsid]), self.sample_counts[dsid]))
            else:
                self.current_indices[dsid] = list(range(floor((1-self.train_split)*self.sample_counts[dsid])))
            if self.shuffle_batch:
                random.shuffle(self.current_indices[dsid])
        self.total_samples = sum([len(self.current_indices[dsid]) for dsid in self.current_indices.keys()])
    
    def get_total_samples(self):
        return self.total_samples
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch with proportional sampling
        
        Returns:
        - Batch of samples
        - Corresponding labels (DSIDs)
        """
        batch_samples = []
        batch_weight_mult_factors = []
        
        # Compute samples per class for this batch
        samples_per_class = {
            dsid: min(max(1, ceil(self.batch_size * prop)), len(self.current_indices[dsid]))
            for dsid, prop in self.class_proportions.items()
        }
        
        # Ensure batch size matches requested size
        total_samples = sum(samples_per_class.values())
        if 0: # Takes all from same one
            if total_samples > self.batch_size:
                # Reduce largest class
                max_dsid = max(samples_per_class, key=samples_per_class.get)
                samples_per_class[max_dsid] -= (total_samples - self.batch_size)
        else:
            while total_samples > self.batch_size:
                # Reduce largest class
                max_dsid = max(samples_per_class, key=samples_per_class.get)
                samples_per_class[max_dsid] -= 1
                total_samples -= 1
            while total_samples < self.batch_size:
                try: 
                    min_dsid = min(
                        (dsid for dsid in self.current_indices.keys() if len(self.current_indices[dsid]) > samples_per_class[dsid]), 
                        key=lambda dsid: samples_per_class[dsid] / self.class_proportions[dsid]
                    )
                except ValueError: # If there aren't any samples left
                    # if sum([len(self.current_indices[dsid])-samples_per_class[dsid] for dsid in self.current_indices.keys()]):
                    break
                samples_per_class[min_dsid] += 1
                total_samples += 1


        
        # Sample from each class
        for dsid, num_samples in samples_per_class.items():
            # Get indices
            indices = self.current_indices[dsid][:num_samples]
            
            # Remove used indices
            self.current_indices[dsid] = self.current_indices[dsid][num_samples:]
            
            # # Replenish indices if depleted
            # if not self.current_indices[dsid]:
            #     self.current_indices[dsid] = list(range(self.sample_counts[dsid]))
            #     random.shuffle(self.current_indices[dsid])
            
            # Fetch samples
            class_samples = self.memmaps[dsid][indices]
            batch_samples.append(class_samples)
            
            # Add corresponding labels
            if (dsid>500000) and (dsid<600000):
                if self.signal_reweights is None:
                    batch_weight_mult_factors.append(np.full(num_samples, 1/self.num_signals*self.bkg_weight_sums/self.abs_weight_sums[dsid]))
                else:
                    batch_weight_mult_factors.append(np.full(num_samples, self.signal_reweights[dsid-510115]/sum(self.signal_reweights)*self.bkg_weight_sums/self.abs_weight_sums[dsid]))
            else:
                batch_weight_mult_factors.append(np.full(num_samples, self.weight_sums[dsid]/self.abs_weight_sums[dsid]))
        
        # print(batch_weigzht_mult_factors)
        # Combine and shuffle final batch
        batch_samples = np.concatenate(batch_samples)
        # print(f"{batch_samples.shape=}")
        batch_weight_mult_factors = np.concatenate(batch_weight_mult_factors)
        MC_Wts = batch_samples[:,1,0].copy()
        batch_samples[:,1,0] *= batch_weight_mult_factors
        
        # Shuffle batch
        shuffled_indices = np.arange(len(batch_samples))
        np.random.shuffle(shuffled_indices)
        batch_samples = batch_samples[shuffled_indices].squeeze()
        MC_Wts = MC_Wts[shuffled_indices]

        # Now extract the training variables, labels, masses, etc.
        # print(batch_samples)
        x = torch.from_numpy(batch_samples[:,2:2+self.objs_to_output,1:])
        x = (x - self.means)/self.stds
        y = torch.nn.functional.one_hot(torch.from_numpy(batch_samples[:,0,0]).to(torch.long), num_classes=self.n_targets).to(torch.float)
        mH = torch.from_numpy(batch_samples[:,0,1])
        mWh_qqbb = torch.from_numpy(batch_samples[:,0,2])
        mWh_lvbb = torch.from_numpy(batch_samples[:,0,3])
        training_Wts = torch.from_numpy(batch_samples[:,1,0])
        dsids = torch.from_numpy(batch_samples[:,1,1])
        MC_Wts = torch.from_numpy(MC_Wts)
        # print(training_Wts)
        training_Wts = np.abs(training_Wts)
        types = torch.from_numpy(batch_samples[:,2:,0]).to(torch.long)
        # print(f"{types.shape=}")

        if self.shuffle_objects:
            if 1: # Switch lepton (1st) and neutrino (2nd), then permute all but the neutrino
                batch_size = x.size(0)
                num_objs = x.size(1)
                inds = torch.empty(batch_size, num_objs).to(torch.long)
                permute_inds = torch.cat((torch.tensor([0]), torch.arange(num_objs-2)+2))#,dim=-1)
                permute_inds
                for i in range(batch_size):
                    inds[i,1:] = permute_inds[torch.randperm(num_objs-1)]
                inds[:,0] = 1
                x=torch.gather(x,1,einops.repeat(inds, 'b o -> b o v',v=x.size(-1)))
                types=torch.gather(types,1,inds)
            elif 1: # Permute all but the first (ie the lepton)
                inds = torch.empty(x.size(0), x.size(1)).to(torch.long)
                for i in range(x.size(0)):
                    inds[i,1:] = torch.randperm(x.size(1)-1)+1
                x=torch.gather(x,1,einops.repeat(inds, 'b o -> b o v',v=x.size(-1)))
                types=torch.gather(types,1,inds)
            elif 1:
                inds = torch.empty(x.size(0), x.size(1)).to(torch.long)
                for i in range(x.size(0)):
                    inds[i] = torch.randperm(x.size(1))
                # x = x[inds]
                # types = types[inds]
                x=torch.gather(x,1,einops.repeat(inds, 'b o -> b o v',v=x.size(-1)))
                # print(f"{types.shape=}")
                # print(f"{inds.shape=}")
                types=torch.gather(types,1,inds)
            else:
                inds = torch.randperm(x.size(1))
                x = x[:,inds]
                types = types[:,inds]
        else:
            # Switch neutrino and lepton, then otherwise leave. 
            # This is so that it can be used consistently with the case where we trained when shuffled
            # and still want it to work when applied to not-shuffled
            # Probably not the most efficient but it's probably fiiiiiine
            batch_size = x.size(0)
            num_objs = x.size(1)
            switched_inds = torch.arange(num_objs).to(torch.long)
            switched_inds[1]=0
            switched_inds[0]=1
            inds = einops.repeat(switched_inds, 'object -> batch object', batch=batch_size)
            x=torch.gather(x,1,einops.repeat(inds, 'b o -> b o v',v=x.size(-1)))
            types=torch.gather(types,1,inds)


        if self.device == 'cuda':
        # if False:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y, types, dsids, training_Wts, mWh_qqbb, mWh_lvbb, MC_Wts, mH = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True), types.pin_memory().to(self.device, non_blocking=True), dsids.pin_memory().to(self.device, non_blocking=True), training_Wts.pin_memory().to(self.device, non_blocking=True), mWh_qqbb.pin_memory().to(self.device, non_blocking=True), mWh_lvbb.pin_memory().to(self.device, non_blocking=True), MC_Wts.pin_memory().to(self.device, non_blocking=True), mH.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y, types, dsids, training_Wts, mWh_qqbb, mWh_lvbb, MC_Wts, mH = x.to(self.device), y.to(self.device), types.to(self.device), dsids.to(self.device), training_Wts.to(self.device), mWh_qqbb.to(self.device), mWh_lvbb.to(self.device), MC_Wts.to(self.device), mH.to(self.device)
        
        return {'x':x, 
                'y':y, 
                'train_wts':training_Wts, 
                'types':types, 
                'dsids':dsids,
                'mWh_qqbb':mWh_qqbb, 
                'mWh_lvbb':mWh_lvbb, 
                'MC_Wts':MC_Wts,
                'mH':mH,
        }

    def __len__(self):
        """
        Calculate total number of complete batches possible
        
        Returns:
        - Number of complete batches that can be generated
        """
        if 0: # Think this is wrong
            # Compute total samples in smallest class
            min_samples = min(len(indices) for indices in self.current_indices.values())
            
            # Compute batches based on proportional sampling
            # return min_samples // self.batch_size
        else:
            all_samples = sum(len(indices) for indices in self.current_indices.values())
            print('WARNING This is kinda of a sketch way of calculating the number of batches.')
            return all_samples // self.batch_size


# %%
# SHUFFLE_OBJECTS = True
# N_TARGETS = 3 # Number of target classes (needed for one-hot encoding)
# N_CTX = 6 # the five types of object, plus one for 'no object;. We need to hardcode this unfortunately; it will depend on the preprocessed root files we're reading in.
# BIN_WRITE_TYPE=np.float32
# max_objs_in_memmap = 8 # BE CAREFUL because this might change and if it does you ahve to rebinarise
# DATAPATH = "/data/atlas/baines/tmp_" + SHUFFLE_OBJECTS*"shuffled_" + f"{max_objs_in_memmap}/"
# N_Real_Vars=4 # x, y, z, energy, d0val, dzval.  BE CAREFUL because this might change and if it does you ahve to rebinarise

# memmap_paths = {}
# for file_name in os.listdir(DATAPATH):
#     if 'shape' in file_name:
#         continue
#     dsid = file_name[5:11]
#     memmap_paths[int(dsid)] = DATAPATH+file_name
#     # print(dsid)
#     # print(file_name)
# class_proportions = None
# batch_size = 256
# %%
# path='/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/20241023_HpWh_TransformerCode/tmp_shuffled_8/dsid_510123.memmap'
# path='/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/20241023_HpWh_TransformerCode/tmp_shuffled_8/dsid_510121.memmap'
# data = np.memmap(
#     path, 
#     dtype=BIN_WRITE_TYPE,  # adjust dtype as needed
#     mode='r', 
#     # shape=(total_samples, -1)  # adjust shape as per your data
#     shape=(total_samples, max_objs_in_memmap+1, N_Real_Vars+1)  # adjust shape as per your data
# )


# %%

# dl = ProportionalMemoryMappedDataset(memmap_paths, class_proportions, batch_size, is_train=False)

# %%
