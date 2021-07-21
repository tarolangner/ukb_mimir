import os
import sys

import time
import numpy as np

import torch
from torch.utils import data
from torch.autograd import Variable
import torchvision

import compressDicom

def main(argv):

    # Where to store results
    path_out = "inference_results/inference_moduletest/"

    # Input parameters
    path_ids = "input_ids.txt" # List of subject ids
    path_dicom_prefix = "/media/veracrypt1/UKB_DICOM/" # Path to DICOM files

    # Get individual input DICOM paths
    paths_dicoms = getDicomPaths(path_ids, path_dicom_prefix, "_20201_2_0.zip")

    # Where to store 2d representations
    path_cache = "cached_images/"

    # Inference batch size
    B = 16

    # Which modules to apply for inference
    paths_modules = [
        "modules/module_test/"
        #"modules/module_organs/",
        #"modules/module_bodycomp/",
        #"modules/module_age/",
        #"modules/module_experimental/"
    ]

    # Run inference
    infer(paths_dicoms, path_cache, paths_modules, path_out, B)


##
# For all chosen DICOMs:
# -Check if 2d representations already exist at cache path
# -For each subject with missing 2d image:
#   -Load DICOM data
#   -Fuse to volume (water and fat signal)
#   -Generate 2d projections and fat fraction slice
#   -Store to cache path
# -Apply inference modules to all subjects
def infer(paths_dicoms, path_cache, paths_modules, path_out, B):

    N = len(paths_dicoms)

    print("##### MIMIR")
    time_start = time.time()

    if os.path.exists(path_out):
        print("ABORT: Output folder already exists at {}".format(path_out) )
        sys.exit()
    else:
        os.mkdir(path_out)

    print("# Pre-processing data")
    paths_img = prepareCache(paths_dicoms, path_cache)

    print("# Applying inference modules")
    applyModules(paths_img, path_cache, paths_modules, path_out, B)

    time_end = time.time()
    print("Total elapsed time: {}".format(time_end - time_start))

    
# Apply each module for inference to all 2d images
def applyModules(paths_img, path_cache, paths_modules, path_out, B):

    time_start = time.time()

    for i in range(len(paths_modules)):

        path_out_i = path_out + os.path.basename(os.path.dirname(paths_modules[i]))
        if not os.path.exists(path_out_i): os.makedirs(path_out_i)

        print("Applying inference module from {}".format(paths_modules[i]))
        print("    Initializing network...")
        net = loadModuleNet(paths_modules[i])

        # Neural network inference
        print("    Performing inference on {} images...".format(len(paths_img)))
        (net_means, net_vars) = applyNetwork(net, paths_img, B)
        del net

        # Post-processing
        print("    Post processing predictions...".format(len(paths_img)))
        postProcessAndWrite(net_means, net_vars, paths_modules[i], paths_img, path_out_i)

        # Write output
        #writePredictions(paths_img, net_means, net_vars, paths_modules[i], path_out_i)

    time_end = time.time()
    print("    Elapsed time: {}".format(time_end - time_start))


# Revert standardization, correct scale and calibrate
def postProcessAndWrite(net_means, net_vars, path_module, paths_img, path_out):

    #
    ## Parse standardization parameters
    path_stand = path_module + "/standardization_parameters.txt"
    with open(path_stand) as f: entries = f.readlines()
    entries.pop(0)

    # Get original mean and standard deviation of training data
    stand_means = np.array([f.split(",")[0] for f in entries]).astype("float")
    stand_stdvs = np.array([f.split(",")[1] for f in entries]).astype("float")

    (N, T) = net_means.shape

    # Revert standardization 
    for t in range(T):
        net_means[:, t] = net_means[:, t] * stand_stdvs[t] + stand_means[t]
        net_vars[:, t] = net_vars[:, t] * np.square(stand_stdvs[t]) # Square for variance from stdev


    #
    ## Apply factors for calibration correction
    path_cal = path_module + "calibration_factors.txt"
    with open(path_cal) as f: entries = f.readlines()
    entries.pop(0)

    # Get factors for scaling of predicted variances
    calibration_factors = np.array([f.split(",")[0] for f in entries]).astype("float")

    # Scale variances thus that the uncertainty estimates are calibrated
    for t in range(T):
        net_vars[:, t] = net_vars[:, t] * calibration_factors[t]


    #
    ## Parse target metadata
    path_meta = path_module + "metadata.txt"
    with open(path_meta) as f: entries = f.readlines()
    entries.pop(0)

    target_names = [f.split(",")[0] for f in entries]
    target_fields = [f.split(",")[1] for f in entries]
    target_units = [f.split(",")[2] for f in entries]
    target_divisors = [f.split(",")[3] for f in entries]

    # Scale according to metadata divisors (converting from mL to L etc)
    target_divisors = np.array(target_divisors).astype("float")
    for t in range(T):
        net_means[:, t] /= target_divisors[t]
        net_vars[:, t] /= target_divisors[t]


    #
    ## Write
    names = [os.path.basename(f).replace(".npy", "") for f in paths_img]

    print("    Writing predictions to {}...".format(path_out))
    with open(path_out + "/predictions.csv", "w") as f:

        # Header
        f.write("name")
        for t in range(T):
            f.write(",{}_mean_in_{}".format(target_names[t], target_units[t]))
            f.write(",{}_variance_in_{}".format(target_names[t], target_units[t]))
        f.write("\n")

        # For each image
        for n in range(N):
            f.write("{}".format(names[n]))

            for t in range(T):
                f.write(",{}".format(net_means[n, t]))
                f.write(",{}".format(net_vars[n, t]))
        
            f.write("\n")


# Use the mean-variance regression network to make predictions
# from the 2d images
def applyNetwork(net, paths_img, B):

    # 
    T = int(net.fc.out_features / 2) # Number of regression targets
    N = len(paths_img) # Number of samples

    # DataLoader for 2d images
    loader = getMipLoader(paths_img, B)
    device = torch.device("cuda:0")

    #
    net = net.to(device)
    net.eval()

    # Reserve for predicted means and predicted variances.
    # Shape is (number of samples, number of targets)
    net_means = np.zeros((N, T))
    net_vars = np.zeros((N, T))

    #
    idx_start = 0
    idx_end = 0

    #
    with torch.no_grad():

        for X in loader:

            # Predict for batch
            X = X.to(device)
            out = net(X)

            # Convert to (B, T, 2), with the latter 
            # being predicted mean and variance
            out = out.view(out.size(0), int(out.size(1)//2), 2)

            # Transform log(variances) to variances
            out[:, :, 1] = torch.exp(out[:, :, 1])

            # Write batch results to output vectors
            idx_end = idx_start + out.size(0)

            #
            out = out.cpu().data[:].numpy()
            net_means[idx_start:idx_end, :] = out[:, :, 0]
            net_vars[idx_start:idx_end, :] = out[:, :, 1]

            idx_start = idx_end

    del loader

    # Return predictions 
    # (Note that these are still in standardized space)
    return (net_means, net_vars)


# DataLoader for 2d MIPs 
# (mean intensity projection images with fat fraction slices)
def getMipLoader(paths_img, B):

    dataset = MipDataset(paths_img)

    params = {"batch_size": B,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True}

    loader = data.DataLoader(dataset, **params)

    return loader


class MipDataset(data.Dataset):

    def __init__(self, paths_img):
        self.paths_img = paths_img

    def __len__(self):
        return len(self.paths_img)

    def __getitem__(self, index):

        # Select sample
        path_img = self.paths_img[index]

        # Load npy
        img = np.load(path_img)
        X = Variable(torch.from_numpy(img)).float()

        return X


def loadModuleNet(path_module):

    # Get number of targets from metadata
    path_meta = path_module + "/metadata.txt"
    with open(path_meta) as f: entries = f.readlines()
    T = len(entries) - 1

    # Initialize network
    net = torchvision.models.resnet50(pretrained=False)
    net.fc = torch.nn.Linear(2048, T*2) # T*2 for mean-variance

    # Load snapshot
    snapshot = torch.load(path_module + "/snapshot.pth.tar", map_location={"cuda" : "cpu"})
    net.load_state_dict(snapshot['state_dict'])
    net.eval()

    print("    Loaded ResNet50 with {} targets".format(T))

    return net


# Check if 2d images exist for all subjects
# For those missing, create new ones from DICOM
def prepareCache(paths_dicoms, path_cache):

    time_start = time.time()

    # Extract DICOM file names
    names = [os.path.basename(f).split(".")[0] for f in paths_dicoms]

    paths_uncached = []

    # If no cache exists, all are uncached
    if not os.path.exists(path_cache): 
        os.mkdir(path_cache)
        paths_uncached = paths_dicoms

    else:
        # Otherwise, check if any are missing
        for i in range(len(paths_dicoms)):
            if not os.path.exists(path_cache + names[i] + ".npy"): 
                paths_uncached.append(paths_dicoms[i])

    #
    names_failed = []
    if not paths_uncached:
        print("    All {} input DICOMs are already cached at \"{}\"".format(len(paths_dicoms), path_cache))
        
    else:
        print("    Of {} input DICOMs, {} are not yet cached at \"{}\"".format(len(paths_dicoms), len(paths_uncached), path_cache))
        print("    This will require an additional {0:0.1f} MB".format(len(paths_uncached) * 0.2))

        print("    Caching missing images...")

        names_failed = cacheImages(paths_uncached, path_cache)

        if names_failed:
            print("    WARNING: {} DICOMs could not be converted and will be excluded from this run".format(len(names_failed)))
            
    # Get paths to all DICOMs that have been successfully compressed
    paths_img = [path_cache + f + ".npy" for f in names if f not in names_failed]

    time_end = time.time()
    print("    Elapsed time: {0:0.1f} s".format(time_end - time_start))

    return paths_img


# DataLoader for compressed UK Biobank DICOM
def getDicomLoader(paths_dicom):

    dataset = compressDicom.DicomDataset(paths_dicom)

    loader = torch.utils.data.DataLoader(dataset,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2**32 + id) )

    return loader


# Load specified DICOMs and compress volumes to 2d
def cacheImages(paths_uncached, path_cache):

    loader = getDicomLoader(paths_uncached)

    N = len(paths_uncached)
    i = 0

    names_failed = []

    #
    for mip_out, dicom_path, error_code in loader:

        i += 1
        name = os.path.basename(dicom_path[0]).split(".")[0]

        if error_code[0]:
            names_failed.append(name)
            print("        Something went wrong with subject {}".format(name))
        else:
            print("        Compressed image {0} ({1:0.3f}% done)".format(i, 100* i / N))
            np.save(path_cache + name + ".npy", mip_out.numpy()[0])


    return names_failed


def getDicomPaths(path_ids, path_dicom_prefix, dicom_suffix):

    # Read ids
    with open(path_ids) as f: entries = f.readlines() 
    entries.pop(0)

    # Convert to DICOM paths
    ids = [f.split(",")[0].split("/")[-1].split(".")[0] for f in entries]
    paths_dicoms = [path_dicom_prefix + f.replace("\n", "") + dicom_suffix for f in ids]

    #paths_dicoms = paths_dicoms[:100] # select only first 100 for testing

    return paths_dicoms


if __name__ == '__main__':
    main(sys.argv)
