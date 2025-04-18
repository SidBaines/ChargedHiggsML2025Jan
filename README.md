# Charged Higgs Transformer code
Code for training, interpreting & applying transformer-like neural networks to low-level reconstructed objects in particle physics events, for event reconstruction/classification.

## Main run files
Note all of these are split into jupyter cell tags (esp. useful for mech interp script)
To train a reconstruction network (on signal only), `python TrainLowLevelReconstruction.py`
To train a classification network (for signal1 vs signal2 vs background), `python TrainLowLevelClassifier.py`
To play around with some interpretability, `python RunLowLevelInterp.py`


## Other run files
Important files (in ~chronological workflow order):
- `cpp_code/` folder contains code for processing ROOT files with initial cuts, decorating the data with the old classificaitons and publication neural network scores, etc.
- `other-python-scripts/preprocessLowLevel.py` **prepares data** for the low-level networks. Contains code for reading ROOT files, doing any necessary samples filtering/resampling for a (set of) training runs & preparing data into numpy .bin format, with shape [batch object variable]. Lots of options for different selections, variables to write, etc. This is primarily used to prepare signal-only datasets for training reconstruction networks, or to prepare signal-and-background datasets for training classification networks, *if* the old reconstruciton method is to be used.
- `other-python-scripts/preprocessHighLevel.py` **prepares data** for the high-level networks. Broadly does the same thing as above, but writes a [batch variable] tensor of *high-level* variables per event. This is used to prepare signal-and-background datasets for training basic MLP classification networks (baseline analysis method)
- `other-python-scripts/preprocessLowLevelApplyRecoSplit.py` **prepares data** for the low-level classifier network, assuming we use reconstruction network. Essentially does the data selection & preparation again, but now also applies the reconstruction network to split into the two possible decay channels based on this. We end up with non-overlapping datasets for each of qqbb & lvbb.
- `other-python-scripts/TrainLowLevelClassifierOldReco.py` **trains** a low-level classifier, assuming that the events are reconstructed using the *old* method.
- `other-python-scripts/ApplyRecoAndClassifiersToRoot.py` applies these networks and rewrites to root files for further analysis

## Utils files
- `utils.py` contains functions which are generally useful across all files
- `lowlevelrecometrics.py` contains a class & methods for tracking performance of the low-level networks for event reconstruction
- `lowlevelmetrics.py` contains a class & methods for tracking performance of the low-level networks for classification (sig1 vs sig2 vs bkg)
- `other-python-scripts/highlevelmetrics.py` contains a class & methods for tracking performance of the high-level networks (sig vs bkg, assuming already split into sig1 vs sig2 during reconstruction)
- `lowleveldataloader.py` contains the dataloader code for reading in preprocessed low-level particle information data for training, applying, or interpreting
- `other-python-scripts/highleveldataloader.py` contains the dataloader code for reading in preprocessed high-level event data for training, applying, or interpreting
- `mechinterputils.py` contains all the useful functions for trying to do mech interp on the low-level info networks