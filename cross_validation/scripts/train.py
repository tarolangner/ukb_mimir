import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import sys
import os
import glob
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data as data

import dataLoading

from torchvision import models


def getPretrainedResnet(T, channel_count, dim):

    # Load pretrained resnet50 and replace final fully-connected layer
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, T * 2) # T * 2 for mean and variance

    return net


def runTraining(target_paths, train_subsets, aug_t, I, checkpoint_path, save_step, B, R, imageset_name, I_lower_lr):

    T = len(target_paths)

    # Get image dimensions and initialize neural network
    (channel_count, dim) = dataLoading.inferImageMetrics(imageset_name)
    net = getPretrainedResnet(T, channel_count, dim)

    # Load dataset of standardized target values
    (dataset, slice_names, [stand_means, stand_stdevs]) = dataLoading.getDataset(target_paths, train_subsets, aug_t, imageset_name, stand=None)

    # Write standardization parameters
    with open(checkpoint_path + "stand.txt", "w") as f:
        f.write("stand_mean,stand_stdev\n")
        for t in range(T): f.write("{},{}\n".format(stand_means[t], stand_stdevs[t]))

    # Create data Loader
    params = {"batch_size": B,
              "shuffle":True,
              "num_workers": 8,
              "pin_memory": True, 
              # use different random seeds for each worker
              # courtesy of https://github.com/xingyizhou/CenterNet/issues/233
              "worker_init_fn" : lambda id: np.random.seed(torch.initial_seed() // 2**32 + id) 
              }

    loader = data.DataLoader(dataset, **params)
        
    # Initiate training
    N = len(slice_names)
    train(net, loader, I, checkpoint_path, save_step, N, B, R, I_lower_lr)


def train(net, loader, I, checkpoint_path, save_step, N, B, R, I_lower_lr):

    #
    torch.backends.cudnn.benchmark = True
    net.train(True)

    #
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=R) 

    # Get number of required epochs
    batch_count = N / float(B)
    E = int(np.ceil(I / batch_count))

    print("Starting training for {} epochs ({} iterations on {} batches)".format(E, I, batch_count))

    start_time = time.time()
    i = 0

    # 
    for e in range(E):

        for X, Y in loader:

            # Skip remains of last epoch
            if i >= I: continue

            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)

            #
            optimizer.zero_grad()
            output = net(X)

            # Calculate loss
            loss = gaussLossWithGaps(output, Y)
            print(loss.item())

            # Perform update
            loss.backward()
            optimizer.step()

            # Save checkpoints
            if i > 0 and ((i+1) % save_step) == 0:
            
                print("Storing snapshot {} of {}".format(int((i+1)/save_step), int(I / save_step)))

                state = {
                    'iteration': i+1,
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict()}

                torch.save(state, checkpoint_path + "snapshot_{}.pth.tar".format(i+1))

                if i == I_lower_lr - 1:
                 
                    # Reduce learning rate for last stages of training
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10

            i += 1
            
    del net
    del loader

    end_time = time.time()
    print("Elapsed training time: {}".format(end_time - start_time))


def mseLossWithGapsOld(output, Y):
    
    # Replace missing gt with predictions
    mask = torch.isnan(Y)
    Y[mask] = output[mask]

    return F.mse_loss(output, Y)


def gaussLossWithGapsNonLogVar(output, Y):

    B = output.shape[0]
    T = int(output.shape[1] / 2)

    output = output.view((B, T, 2))

    # Replace missing gt with predictions
    mask = torch.isnan(Y)
    Y[mask] = output[:, :, 0][mask]

    softplus_var = torch.log(1 + torch.exp(output[:, :, 1])) + 0.000001
    loss = torch.log(softplus_var) / 2 + torch.square(Y - output[:, :, 0]) / (2*softplus_var) 

    loss[mask] = 0

    loss = torch.mean(loss)

    return loss


def mseLossWithGaps(output, Y):

    B = output.shape[0]
    T = int(output.shape[1] / 2)

    output = output.view((B, T, 2))

    mean = output[:, :, 0]
    log_var = output[:, :, 1]

    # Replace missing gt with predictions
    mask = torch.isnan(Y)
    Y[mask] = mean[:, :][mask]

    loss = torch.pow(Y - mean, 2)
    loss[mask] = 0 # remove influence of missing gt

    loss = torch.mean(loss)

    return loss


# Note: In this formulation, the output in view (B, T, 2)
# contains as last dimension the mean and log(variance).
# Courtesy of Fredrik K Gustafsson
def gaussLossWithGaps(output, Y):

    B = output.shape[0]
    T = int(output.shape[1] / 2)

    output = output.view((B, T, 2))

    mean = output[:, :, 0]
    log_var = output[:, :, 1]

    # Replace missing gt with predictions
    mask = torch.isnan(Y)
    Y[mask] = mean[:, :][mask]

    loss = torch.exp(-log_var) * torch.pow(Y - mean, 2) + log_var
    #loss = torch.exp(-log_var) * torch.abs(Y - mean) + log_var # LAPLACE
    loss[mask] = 0 # remove influence of missing gt

    loss = torch.mean(loss)

    return loss


