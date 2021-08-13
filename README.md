# MIMIR
**An Inference Engine for UK Biobank Neck-to-knee Body MRI**

*Note: This repository is still under development*

This repository implements an experimental software for fully automated analysis of magnetic resonance images (MRI) of the [UK Biobank](https://www.ukbiobank.ac.uk/).   [***M**edical **I**nference on **M**agnetic resonance images with **I**mage-based **R**egression* (MIMIR)](https://arxiv.org/abs/2106.11731) as implemented here utilizes convolutional neural networks for mean-variance regression.

Learn more about [the UK Biobank Imaging Study](https://www.nature.com/articles/s41467-020-15948-9) and the [uncertainty-aware deep regression method implemented here](https://arxiv.org/abs/2101.06963).


---
### Basic FAQ

1) *What does it do?*

This PyTorch implementation can be applied to neck-to-knee body MRI of UK Biobank to estimate and emulate a wide range of measurements. Note that this experimental software is used at your own risk, is provided with no guarantees of anything, and is not a certified medical diagnostic tool.

2) *Which properties can the inference predict?*

This inference engine can estimate sex, age, height, weight, and several emulated measurements together with individual confidence intervals. Find a full list of regression targets and results of 10-fold cross-validation [here](https://github.com/tarolangner/ukb_mimir/blob/main/documentation/validation_results/mimir_validation.pdf).

3) *How to get image data?*

The image data of UK Biobank can only be shared with authorized research applications. You can [apply for access here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). Within UK Biobank, these images are listed under field *20201 - Dixon technique for internal fat - DICOM*.
If you do not have access but wish to test the pipeline, you can apply it to one test image that is [freely available here](https://biobank.ndph.ox.ac.uk/showcase/refer.cgi?id=20201).

4) *Will this work for any medical image data?*

The approach has only been validated for neck-to-knee body MRI of about 40,000 UK Biobank subjects (44-82 years old, 95% self-reported white British ethnicity). Data from other imaging modalities, device types, or demographics will likely result in deteriorated performance. However, reproducing the imaging protocol on similar demographics should enable viable results.

5) *How does it work?*

In a nut shell, the inference engine compresses the volumetric MRI to a 2d format and applies ResNet50 instances to it, which predict both the mean and variance of each given measurement for any given subject. They predict the target values and an estimate of predictive uncertainty, which yields a confidence interval. Find user instructions below.

6) *How was it created and validated?*

Code for training and evaluating the underlying code is provided in the *cross_validation* sudirectory, together with additional documentation.

7) *I have technical/ethical/spiritual concerns and wish to speak to the manager*

You can try the contact details listed at the end of this file.

---

### Instructions:

1) Download the inference modules from [the shared folder at Uppsala Sunet Box](https://uppsala.box.com/s/k04jl8npr3792urscue2u4ov47jmeahb).
Unpack them into "modules/".

2) Optionally, download UK Biobank Return dataset 3072 for annotations of the UK Biobank neck-to-knee body MRI. It contains annotations for all those cases with artefacts that should be excluded.

3) In *mimirInference.py* call *infer* with:
    * A list of paths to the UK Biobank neck-to-knee body MRI dicom files (field 20201)
    * A path to a temporary caching folder for the 2d representations
    * A list of paths to the modules to be applied 
    * An output path for the csv files with predictions
    * The batch size B (set as high as your GPU allows for fastest speed)

4) Find the predictions in the output folder. The predicted variances can be used to calculate confidence intervals.

---

For any questions and feedback, feel free to contact taro.langner(at).surgsci.uu.se
