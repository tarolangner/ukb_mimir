# MIMIR
*An Inference Engine for UK Biobank Neck-to-knee Body MRI*

**Note: This repositry is still work in progress**

This repository implements an experimental software for fully automated analysis of magnetic resonance images (MRI) of the UK Biobank study. The ***M**edical **I**nference on **M**agnetic resonance images with **I**mage-based **R**egression* (MIMIR) is enabled by convolutional neural networks for mean-variance regression.

Learn more about [the UK Biobank Imaging Study](https://www.nature.com/articles/s41467-020-15948-9) and the [Uncertainty-aware deep regression method](https://arxiv.org/abs/2101.06963).

By deploying the PyTorch code of this repository, a wide range of metadata and emulated measurements can be inferred from one or more DICOM files with MRI data of the neck-to-knee body MRI imaging protocol as used by UK Biobank. Note that this experimental software is used at your own risk, is provided with no guarantees of anything, and is not a certified medical diagnostic tool.


---
### Basic FAQ

1) *Which measurements does the inference engine predict?*

It can estimate sex, age, height, weight and several emulated measurements together with confidence intervals. Find a full list of regression targets and results of 10-fold cross-validation here (TODO: LINK). 

2) *How to get image data?*

The image data of UK Biobank can only be shared with authorized research applications. You can apply for access [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). Within UK Biobank, these images are listed under field *20201 - 	Dixon technique for internal fat - DICOM*.

3) *Will this work for any medical image data?*

The approach has only been validated for neck-to-knee body MRI of about 40,000 UK Biobank subjects (44-82 years old, 95% self-reported white British ethnicity). Data of a different imaging modality, device type, or demographic will likely result in deteriorated performance. However, if the imaging protocol was accurately replicated outside the scope of UK Biobank for a similar demographic, similar performance might be possible.

4) *How does it work?*

In a nut shell, the inference engine compresses the volumetric MRI to a 2d format and applies ResNet50 instances to it, which predict both the mean and variance of each given measurement for any given subject. They predict the target values and an estimate of predictive uncertainty, which yields a confidence interval. Find user instructions below.

5) *How was it created and validated?*

Additional documentation and code will be provided in the *training* subdirectory.

6) *I have technical/ethical/spiritual complaints and want to speak to the manager"

You can try the contact details listed at the end of this file.

---

### Instructions:
-TODO, modules

---

For any questions and feedback, feel free to contact taro.langner(at).surgsci.uu.se
