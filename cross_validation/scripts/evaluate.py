import cv2
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
import shutil


def evaluate(net, loader, B, N):

    net = net.cuda()
    net.eval()

    #  
    values_out_means = [] # estimated means
    values_out_vars = [] # estimated variances
    values_gt = [] # ground truth values

    i_start = 0

    T = None

    #
    for X, Y in loader:

        # Lazy initialize target count
        if T is None:   
            T = Y.size(1)
            values_out_means = np.zeros((N, T))
            values_out_vars = np.zeros((N, T))
            values_gt = np.zeros((N, T))

        #
        X = X.cuda(non_blocking=True)
        output = net(X)

        # Calculate effective batch size
        B_i = B
        i_end = i_start + B_i

        # Last batch may wrap over, so only use unique entries
        if i_end > N:
            B_i = N % B
            i_end = i_start + B_i

        # Reshape output. Last dimension now contains
        # subject- and target-wise estimated mean and log variance
        output = output.view((B_i, T, 2))

        # Convert log variance returned by network to variance.
        # Courtesy of Fredrik K Gustafsson
        output[:, :, 1] = torch.exp(output[:, :, 1]) 

        # Move to cpu
        out = output.cpu().data[:].numpy()

        values_out_means[i_start:i_end, :] = out[:B_i, :, 0]
        values_out_vars[i_start:i_end, :] = out[:B_i, :, 1]
        values_gt[i_start:i_end, :] = Y[:B_i, :]

        i_start = i_end

    return (values_out_means, values_out_vars, values_gt)
