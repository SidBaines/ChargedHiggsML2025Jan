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
from copy import copy

DSID_TO_POLE_MASS = {
    510115: 0.8,
    510116: 0.9,
    510117: 1.0,
    510118: 1.2,
    510119: 1.4,
    510120: 1.6,
    510121: 1.8,
    510122: 2.0,
    510123: 2.5,
    510124: 3.0,
}
ALL_SIGNAL_MASSES = torch.tensor(list(DSID_TO_POLE_MASS.values()))

class ProportionalMemoryMappedDatasetHighLevel:
    def __init__(self, 
                 memmap_paths: Dict[int, str],  # DSID to memmap path
                 N_Real_Vars: int = 7,
                 class_proportions: Dict[int, float] = None,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 is_train: bool = True,
                 validation_split_idx: int = 0,
                 n_splits: int = 2,
                 n_targets: int = 3,
                 shuffle_batch: bool = True, # Whether or not to shuffle along the batch dimension, should be true unless we're testing
                 signal_reweights: typing.Optional[np.array]=None,
                 means=None,
                 stds=None,
                 has_eventNumbers=False,
                 return_pole_mass=False,
                 ):
        """
        Initialize a dataset loader with memory-mapped files for each class (DSID)
        
        Args:
        - memmap_paths: Dictionary mapping DSID to path of memory-mapped file
        - class_proportions: Optional dictionary of sampling proportions for each DSID
        - batch_size: Number of samples per batch
        """
        # Load memory-mapped files
        self.memmaps = {}
        self.sample_counts = {}
        self.abs_weight_sums = {}
        self.weight_sums = {}
        self.device = device
        self.n_targets = n_targets
        self.signal_reweights = signal_reweights
        self.shuffle_batch = shuffle_batch
        self.has_eventNumbers = has_eventNumbers
        self.return_pole_mass = return_pole_mass

        # Add train/val splitting logic
        self.is_train = is_train
        self.validation_split_idx = validation_split_idx
        self.n_splits = n_splits
        # Add/populate with defaults the mean/std for scaling inputs
        if means is not None:
            self.means = torch.Tensor(means).to(torch.float32)
        else:
            self.means = torch.zeros(N_Real_Vars).to(torch.float32)
        if stds is not None:
            self.stds = torch.Tensor(stds).to(torch.float32)
        else:
            self.stds = torch.ones(N_Real_Vars).to(torch.float32)

        self.bkg_weight_sums = 0
        # self.sig_weight_sums = 0
        self.num_signals = 0
        self.signal_masses = []
        self.signal_dsids = []

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
                    shape=(total_samples, N_Real_Vars+6)  # adjust shape as per your data
                )
                self.abs_weight_sums[dsid] = sum_abs_weights
                self.weight_sums[dsid] = sum_weights
                if (dsid<500000) or (dsid > 600000):
                    self.bkg_weight_sums += sum_weights
                else:
                    # self.sig_weight_sums += sum_weights
                    self.num_signals += 1
                    self.signal_masses.append(DSID_TO_POLE_MASS[dsid])
                    self.signal_dsids.append(dsid)
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
            if self.has_eventNumbers:
                if self.is_train:
                    # self.current_indices[dsid] = np.arange(self.sample_counts[dsid])[~(np.isin((self.memmaps[dsid][:,1,2] % self.n_splits), self.validation_split_idx))]
                    self.current_indices[dsid] = np.arange(self.sample_counts[dsid])[(self.memmaps[dsid][:,5] % self.n_splits) != self.validation_split_idx]
                else:
                    # self.current_indices[dsid] = np.arange(self.sample_counts[dsid])[(np.isin((self.memmaps[dsid][:,1,2] % self.n_splits), self.validation_split_idx))]
                    self.current_indices[dsid] = np.arange(self.sample_counts[dsid])[(self.memmaps[dsid][:,5] % self.n_splits) == self.validation_split_idx]
            else:
                if self.is_train:
                    self.current_indices[dsid] = np.arange(self.sample_counts[dsid])[(np.arange(self.sample_counts[dsid]) % self.n_splits) != self.validation_split_idx]
                else:
                    self.current_indices[dsid] = np.arange(self.sample_counts[dsid])[(np.arange(self.sample_counts[dsid]) % self.n_splits) == self.validation_split_idx]
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
        # print("NEED TO CHECK YIELDS OF LVBB/QQBB AND COMPARE THE DATASET WRITING/READING TO THE LOW-LEVEL ONE BECAUSE THIS DOESN'T SEEM TO MATCH AT THE MOMENT")
        # assert(False)
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
        MC_Wts = batch_samples[:,1].copy()
        # print(batch_samples[:,1].sum())
        batch_samples[:,1] *= batch_weight_mult_factors
        # print(batch_samples[:,1].sum())
        
        # Shuffle batch
        if self.shuffle_batch:
            shuffled_indices = np.arange(len(batch_samples))
            np.random.shuffle(shuffled_indices)
        else:
            shuffled_indices = np.arange(len(batch_samples))
        batch_samples = batch_samples[shuffled_indices].squeeze()
        MC_Wts = MC_Wts[shuffled_indices]
        # print(batch_samples[:,1].sum())

        
        # Now extract the training variables, labels, masses, etc.
        y = torch.nn.functional.one_hot(torch.Tensor(batch_samples[:,0]).to(torch.long), 3).to(torch.float)
        # Convert from {0:bkg, 1:lvbb, 2:qqbb} to {0:bkg, 1:sig} since we only have one type of signal here
        if (y[:,1].sum().item() == 0):
            y = y[:,[0,2]]
        elif (y[:,2].sum().item() == 0):
            y = y[:,[0,1]]
        assert(y.shape[1]==self.n_targets)
        training_Wts = torch.from_numpy(batch_samples[:,1])
        # print(training_Wts.sum())
        mWh = torch.from_numpy(batch_samples[:,2])
        dsid = torch.from_numpy(batch_samples[:,3])
        mH = torch.from_numpy(batch_samples[:,4])
        if self.has_eventNumbers:
            x = torch.from_numpy(batch_samples[:,6:]) # The index at 5 is the event number for train/val split
            if self.return_pole_mass:
                pole_mass = torch.tensor(self.signal_masses)[(torch.from_numpy(batch_samples[:,5]) % len(self.signal_masses)).to(int)]
                pole_mass[(dsid>=500000) & (dsid<600000)] = ALL_SIGNAL_MASSES[dsid[(dsid>=500000) & (dsid<600000)].to(int)-510115]
            else:
                pole_mass = None
        else:
            x = torch.from_numpy(batch_samples[:,5:])
        x = (x - self.means)/self.stds
        MC_Wts = torch.from_numpy(MC_Wts).reshape(training_Wts.shape)
        # print(training_Wts)
        training_Wts = np.abs(training_Wts)

        if self.device == 'cuda':
        # if False:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y, mWh, dsid, training_Wts, MC_Wts = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True), mWh.pin_memory().to(self.device, non_blocking=True), dsid.pin_memory().to(self.device, non_blocking=True), training_Wts.pin_memory().to(self.device, non_blocking=True), MC_Wts.pin_memory().to(self.device, non_blocking=True)
            if self.return_pole_mass:
                pole_mass = pole_mass.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y, mWh, dsid, training_Wts, MC_Wts = x.to(self.device), y.to(self.device), mWh.to(self.device), dsid.to(self.device), training_Wts.to(self.device), MC_Wts.to(self.device)
            if self.return_pole_mass:
                pole_mass = pole_mass.to(self.device)
        
        ret_dict = {'x':x, 
                'y':y, 
                'mWh':mWh, 
                'dsid':dsid,
                'train_wts':training_Wts, 
                'MC_Wts':MC_Wts,
                'mH':mH,
        }
        if self.return_pole_mass:
            ret_dict['pole_mass'] = pole_mass
        return ret_dict

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
            return ((all_samples-1) // self.batch_size) + 1
