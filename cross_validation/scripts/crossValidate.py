import numpy as np
import time
import sys
import os
import glob
import copy
import shutil

import torch.nn as nn
import torch
import torch.utils.data as data

from torchvision import models

import train
import evaluate
import storeEvaluation
import dataLoading

#
def main(argv):

    output_path = "../networks/Mimir_m4_traintest/"
    imageset_path = "/home/taro/DL_imagesets/UKB_dual_projFF_uint8/"

    if False:
        target_paths = ["../targets/UKB_dual_projFF_uint8_AtlasVisceralAdiposeTissue_10fold/",
                        "../targets/UKB_dual_projFF_uint8_AtlasAbdominalSubcutaneousAdiposeTissue_10fold/",
                        "../targets/UKB_dual_projFF_uint8_AtlasLiverFat_10fold/",
                        "../targets/UKB_dual_projFF_uint8_AtlasThighMuscle_10fold/"]

    if True:
        target_paths = ["../targets/UKB_dual_projFF_uint8_AtlasVisceralAdiposeTissue_traintest/",
                        "../targets/UKB_dual_projFF_uint8_AtlasAbdominalSubcutaneousAdiposeTissue_traintest/",
                        "../targets/UKB_dual_projFF_uint8_AtlasLiverFat_traintest/",
                        "../targets/UKB_dual_projFF_uint8_AtlasThighMuscle_traintest/"]
                        
    print("Found {} targets".format(len(target_paths)))

    I = 10000 # training iterations
    save_step = I / 10 # snapshot saving interval
    I_lower_lr = I / 10 * 8 # iteration after which to reduce learning rate by factor 10

    B = 32 # batch size
    R = 0.00005 # learning rate

    do_train = False
    metrics_only = False

    start_k = 0 # first cross-validation split
    end_k = 1 # last cross-validation split

    # Augmentation parameters
    aug_t = np.array((0.0, 0.0, 16.0, 16.0)) # Maximum translations: C, Z, Y, X

    print("#################### Training network:")
    print(f"Output at {output_path}")

    if metrics_only:
        if not os.path.exists(output_path):
            print("ABORT: Network folder not found, and can therefore not be evaluated")
    #
    if do_train and start_k == 0:

        if os.path.exists(output_path):
            #print("ABORT: Network output already exists!")
            #sys.exit()
            shutil.rmtree(output_path)
                    
        os.makedirs(output_path)
        createDocumentation(output_path, target_paths)

    runCrossValidation(output_path, target_paths, I, save_step, B, R, do_train, metrics_only, start_k, end_k, aug_t, imageset_path, I_lower_lr)
    

def runCrossValidation(output_path, target_paths, I, save_step, B, R, do_train, metrics_only, start_k, end_k, aug_t, imageset_path, I_lower_lr):

    start_time_total = time.time()

    # Get number of subsets from first target
    subset_lists = [f for f in os.listdir(target_paths[0] + "/subsets/") if os.path.isfile(os.path.join(target_paths[0] + "/subsets/", f))]
    subset_lists = [f for f in subset_lists if "_ids.txt" in f]

    # Get cv indices
    K = len(subset_lists)
    cv_subsets = range(K)

    # Check if full cross-validation is used
    if K != 2 and (start_k != 0 or end_k != K):
        print("WARNING: Training on subsets {}-{} even though split is {}-fold!".format(start_k, end_k, K))

    #
    if not metrics_only:

        for k in range(start_k, end_k):

            # Validate against subset k
            val_subset = cv_subsets[k]

            # Train on all but subset k
            train_subsets = [x for f,x in enumerate(cv_subsets) if f != k]

            print("########## Validating against subset {}".format(val_subset))

            #
            output_path_k = output_path + "subset_{}/".format(val_subset)
            checkpoint_path = output_path_k + "snapshots/"

            #
            if do_train:

                os.makedirs(output_path_k)
                os.makedirs(checkpoint_path)

                # TRAIN
                start_time = time.time()
                train.runTraining(target_paths, train_subsets, aug_t, I, checkpoint_path, save_step, B, R, imageset_path, I_lower_lr)
                end_time = time.time()

                documentTrainingTime(output_path, (end_time - start_time))

            # EVALUATE 
            runEvaluation(target_paths, val_subset, I, save_step, checkpoint_path, output_path_k, imageset_path)

    #
    if not os.path.exists(output_path + "eval/"):
        os.makedirs(output_path + "eval/")

    end_time_total = time.time()
    storeEvaluation.aggregateValidation(output_path, target_paths, I, save_step, (end_time_total - start_time_total))


def documentTrainingTime(output_path, runtime):

    # Write runtime file
    runtime_file = open(output_path + "runtime.txt","w") 
    runtime_file.write("{}".format(runtime))
    runtime_file.close() 


def readStandardizationParameters(path_stand):

    with open(path_stand) as f: entries = f.readlines()
    entries.pop(0)

    stands = []
    for t in range(len(entries)):
        (stand_mean, stand_stdev) = entries[t].split(",")
        stand = {
            "mean" : stand_mean,
            "stdev" : stand_stdev
            }
        stands.append(stand)

    return stands


## Make predictions for the validation set with all stored snapshots of the given subset,
# store them to file and write evaluation metrics.
def runEvaluation(target_paths, val_index, I, save_step, checkpoint_path, subset_path, imageset_path):

    B = 8 # Batch size for predicting
    T = len(target_paths)

    # Load pretrained resnet50 and replace final fully-connected layer
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, T * 2) # T * 2 for mean and variance

    # Read standardization
    stands = readStandardizationParameters(checkpoint_path + "stand.txt")

    # Get data loader for evaluation
    (dataset, img_names, _) = dataLoading.getDataset(target_paths, [val_index], aug_t=np.zeros(4), images_path=imageset_path, stand=stands)

    if len(img_names) == 0:
        print("Validation set is empty. Concluding training...")
        sys.exit()

    output_path = subset_path + "eval/"
    if not os.path.exists(output_path): os.makedirs(output_path)

    params = {"batch_size": B,
              "shuffle":False,
              "num_workers": 8,
              "pin_memory": True}

    loader = data.DataLoader(dataset, **params)

    # 
    S = int(I // save_step)
    N = len(img_names)

    # Initialize predicted means and variances in (sample, target, snapshot)
    values_out_means = np.zeros((N, T, S))
    values_out_vars = np.zeros((N, T, S))

    # Iniialize ground truth values (sample, target)
    values_gt = np.zeros((N, T))

    # Evaluate all snapshots
    for i in range(S):

        print("Evaluating snapshot {}".format(i+1))

        # Load network weights from snapshot
        snapshot_file = checkpoint_path + "snapshot_{}.pth.tar".format(int((i+1)*save_step))
        
        snapshot = torch.load(snapshot_file, map_location={"cuda" : "cpu"})
        net.load_state_dict(snapshot['state_dict'])

        # Evaluate on validation set
        (values_out_means[:, :, i], values_out_vars[:, :, i], values_gt[:, :]) = evaluate.evaluate(net, loader, B, N)

    del net
    del loader

    # Store predictions and evaluation metrics to file
    storeEvaluation.storePredictions(img_names, values_gt, values_out_means, values_out_vars, I, save_step, output_path, target_paths, stands)


def createDocumentation(network_path, target_paths):

    os.makedirs(network_path + "documentation")
    for file in glob.glob("*.py"): shutil.copy(file, network_path + "documentation/")

    subfolders = ["data", "eval", "models", "training"]

    for s in subfolders:
        
        sub_path = network_path + "documentation/" + s

        os.makedirs(sub_path)
        for file in glob.glob(s + "/*.py"): shutil.copy(file, sub_path)

    #
    with open(network_path + "target_paths.txt", "w") as f:
        for t in range(len(target_paths)):
            f.write("{}\n".format(os.path.abspath(target_paths[t])))
        


if __name__ == '__main__':
    main(sys.argv)
