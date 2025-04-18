# Charged Higgs Transformer code
Code for applying transformer-like neural networks to low-level reconstructed objects in particle physics events, for event reconstruction/classification

## Main run files
Important files (in ~chronological workflow order):
- `cpp_code` folder contains code for processing ROOT files with initial cuts, decorating the data with the old classificaitons and publication neural network scores, etc.
- `preprocessLowLevel` **prepares data** for the low-level networks. Contains code for reading ROOT files, doing any necessary samples filtering/resampling for a (set of) training runs & preparing data into numpy .bin format, with shape [batch object variable]. Lots of options for different selections, variables to write, etc. This is primarily used to prepare signal-only datasets for training reconstruction networks, or to prepare signal-and-background datasets for training classification networks, *if* the old reconstruciton method is to be used.
- `preprocessHighLevel` **prepares data** for the high-level networks. Broadly does the same thing as above, but writes a [batch variable] tensor of *high-level* variables per event. This is used to prepare signal-and-background datasets for training basic MLP classification networks (baseline analysis method)
- `TrainLowLevelReconstruction` **trains** the low-level reconstruction network
- `preprocessLowLevelApplyRecoSplit` **prepares data** for the low-level classifier network, assuming we use reconstruction network. Essentially does the data selection & preparation again, but now also applies the reconstruction network to split into the two possible decay channels based on this. We end up with non-overlapping datasets for each of qqbb & lvbb.
- `TrainLowLevelClassifierOldReco` **trains** a low-level classifier, assuming that the events are reconstructed using the *old* method.
- `TrainLowLevelClassifier` **trains** a low-level classifier, assuming that the events are reconstructed using the *new* method.
- `MechInterpLowLevelReco` attempts to apply some rudimentary mech interp techniques to understand the internal workings of the low-level model (eg. attention analysis, direct logit attribution, symbolic regression of attention outputs/residual stream features)
- `ApplyRecoAndClassifiersToRoot` applies these networks to

## Utils files
- `utils.py` contains functions which are generally useful across all files
- `MetricsLowLevelRecoTruthMatching.py` contains a class & methods for tracking performance of the low-level networks
- `MetricsLowLevel.py` contains a class & methods for tracking performance of the low-level networks
- `MetricsHighLevel.py` contains a class & methods for tracking performance of the high-level networks
- 