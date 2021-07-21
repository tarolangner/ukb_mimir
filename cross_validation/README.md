# MIMIR
**Cross-validation code**

*Note: This repository is still under development*

This directory implements a system for cross-validation of the deep regression networks. It can also be used to train new inference modules for the MIMIR pipeline itself.

---
### Performing a cross-validation

1) *Prepare the image data*
Use *compressDicom.py* from the main directory to extract 2d reprentations of all neck-to-knee body MRI DICOMs that are to be included.
Note: You can simply run the inference pipeline without predicting anything and the images will be stored into the *cached_images* folder.

2) *Create a training-validation split*
Create a directory in *cross_validation/splits/* and add files named "images_set_0.txt" up to "images_set_10.txt" for a 10-fold cross-validation split.
Each file should contain a list of subject IDs (one per line) of subjects to validate against in the given split.

3) *Prepare the regression targets*
Use *scripts/formatTarget.py* to extract and format target values from a UK Biobank field to *cross_validation/targets/*.
Repeat for all desired targets.

4) *Run cross-validation*
Use *crossValidate.py* to perform a cross-validation on the chosen target(s)


---
### Creating a new inference module

1) *Perform cross-validation as described above*
This is needed to determine the scaling factors for the uncertainty calibration

2) *Create train-test split*
Similar to the step 2) above, split the training samples. In order to use all samples for training, list their ids in *images_set_1.txt* and leave *images_set_0.txt* empty.

3) *Combine into inference module*
Use *createInferenceModule.py* to combine the required files. The snapshot should be from the last iteration of the preceding step 2. The calibration factors, in contrast,
should refer to a preceding cross-validation with the same configuration.<sup>1</sup>



---
<sup>1</sup>On the calibration factors: For each prediction, the neural network outputs a variance that also serves as uncertainty estimate. These values are typically too low to accurately describe prediction intervals. The implementation therefore provides target-wise scaling factors, by using a simple annealing search to achieve better calibration on the validation sets in cross-validation. These are assumed to also be approximately valid for a snapshot trained on all training samples, as used for inference modules.
