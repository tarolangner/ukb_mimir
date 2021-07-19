import sys
import os
import numpy as np

import scipy.ndimage

import torch
from torch.autograd import Variable
from torch.utils import data


class ProjectionDataset(data.Dataset):

    def __init__(self, img_paths, labels, aug_t):

        self.labels = labels
        self.img_paths = img_paths
        self.aug_t = aug_t

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        # Select sample
        img_path = self.img_paths[index]

        # Load data and get label
        img = np.load(img_path)

        D = len(img.shape)

        if np.count_nonzero(self.aug_t) > 0:

            # Augment by random translation
            rand = np.random.random(D) 

            disp = np.around((2 * rand - 1) * self.aug_t[-D:])
            img = scipy.ndimage.shift(img, disp, order=0, mode="nearest")

        #
        X = Variable(torch.from_numpy(img)).float()
        Y = Variable(torch.from_numpy(np.array(self.labels[index, :]))).float()

        return X, Y


def parseTarget(target_path, subsets, images_path):

    #
    img_names = []
    img_labels = []

    # Define img_paths
    for k in subsets:
        
        # Load image names for subset
        img_names_file_k = target_path + "subsets/subset_{}_ids.txt".format(k)
        with open(img_names_file_k) as f:
            img_names_k = f.read().splitlines()

        img_label_file_k = target_path + "subsets/subset_{}_labels.txt".format(k)
        with open(img_label_file_k) as f:
            img_labels_k = f.read().splitlines()

        img_names.extend(img_names_k)
        img_labels.extend(img_labels_k)

    img_paths = [images_path + f + ".npy" for f in img_names]
    img_labels = np.array(img_labels).astype("float")

    return (img_paths, img_labels)


## Load all available samples for the specified targets and subsets.
# Each sample represents one subject as:
#    -Image name 
#    -T standardized target values (set to NaN if unknown)
# Return a PyTorch dataset, the union of available image names, and target-wise standardization parameters.
def getDataset(target_paths, subsets, aug_t, images_path, stand=None, ids_train_excl=None):

    #
    img_names_all = []
    img_labels_all = []

    # Load each target
    for t in range(len(target_paths)):

        (img_names_t, img_labels_t) = parseTarget(target_paths[t] + "/", subsets, images_path)

        idx = np.argsort(img_names_t)

        img_names_all.append(np.array(img_names_t)[idx])
        img_labels_all.append(np.array(img_labels_t)[idx].astype("float"))
        print("Found {} samples for target {}".format(len(img_names_t), t))

    # Extract unique ids
    img_names = []
    for t in range(len(img_names_all)): img_names.extend(img_names_all[t])
    img_names = np.sort(np.unique(np.array(img_names)))

    #
    N = len(img_names)
    T = len(target_paths)

    print("{} unique samples in union".format(N))

    # Assign consolidated labels with NaN for gaps
    img_labels = np.ones((N, T))
    img_labels[:] = np.nan

    # Standardization parameters
    label_means = np.zeros(T)
    label_stdevs = np.zeros(T)

    # Standardize target values and copy them to union (with gaps)
    for t in range(T):
        
        if stand is None:

            # Apply standardization
            label_means[t] = np.mean(img_labels_all[t])
            label_stdevs[t] = np.std(img_labels_all[t], ddof=1)

        else:
            # Use already provided standardization
            label_means[t] = float(stand[t]["mean"])
            label_stdevs[t] = float(stand[t]["stdev"])

        # Apply standardization
        img_labels_all[t] = (img_labels_all[t] - label_means[t]) / label_stdevs[t]

        # Copy to with gaps
        mask = np.in1d(img_names, img_names_all[t])
        img_labels[mask, t] = img_labels_all[t]

        if not np.array_equal(img_names[mask], img_names_all[t]):
            print("ERROR: Image names in subset of union do not match target {}".format(t))
            sys.exit()

    dataset = ProjectionDataset(img_names, img_labels, aug_t)

    return (dataset, img_names, [label_means, label_stdevs])


def inferImageMetrics(imageset_path):

    files = [f for f in os.listdir(imageset_path) if os.path.isfile(os.path.join(imageset_path, f))]

    # Load one file
    file_name = files[0]
    image = np.load(imageset_path + file_name)

    dim = image.shape[1:]
    channel_count = image.shape[0]

    return (channel_count, dim)
