# Modules

This directory contains the inference modules. Each module consists of a trained network snapshot, parameters for standardization and calibration, and metadata.

The files are available at a separate download here:  
[Shared folder at Uppsala Sunet Box](https://uppsala.box.com/s/k04jl8npr3792urscue2u4ov47jmeahb)

After downloading, unpack the corresponding .zip into this directory (as in: modules/module_bodycomp/).

## Structure
Each module contains:
* *snapshot.pth.tar* Trained ResNet50 weights for PyTorch
* *calibration_factors.txt* Scaling factors for the predicted variances
* *metadata.txt* For each predicted property, the name, field ID, measurement unit, etc
* *standardization_parameters.txt* The mean and standard deviation of the original values used for training

### module_bodycomp
-TODO

### module_organs
-TODO

### module_age
-TODO

### module_experimental
-TODO


### FAQ

1) *Why is the inference split in modules at all?*
In principle, one network can predict all 72 targets at once. However, the prediction accuracy was found to deteriorate when too many targets were predicted at the same time, especially when age is included. The chosen 

2) *Why not use ensembles?*

