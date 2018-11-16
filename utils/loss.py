import os
import sys
import torch
import timeit
import argparse
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import pdb

def isnan(x):
    return x != x

class loss:

    def MSLoss(self, X_sample, X):
        t = torch.mean(torch.norm((X_sample - X),1),dim=0) 
        return t
    
    def BCELoss(self, y_pred, y, eps = 1e-25):
        # y_pred_1 = torch.log(y_pred+ eps)
        # y_pred_2 = torch.log(1 - y_pred + eps)
        # t = -torch.sum(torch.mean(y_pred_1*y + y_pred_2*(1-y),dim=0))
        t = torch.nn.functional.binary_cross_entropy(y_pred, y)*y.shape[-1]
        if(torch.isnan(t).any()):
            print("nan")
            pdb.set_trace()
        if(t<0):
            print("negative")            
            pdb.set_trace()
        return t
    
    def L1Loss(self, X_sample, X):
        t = torch.sum(torch.mean(torch.abs(X_sample - X),dim=0))
        return t
