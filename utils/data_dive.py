import os
import sys
# import torch
import timeit
import argparse
import numpy as np
import time
# import torch.nn as nn
# import torch.optim as optim
import matplotlib.pyplot as plt
# import torch.autograd as autograd
from sklearn import preprocessing
# from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

# this file is to explore the generated data and the data that already exist to see how much similarity do they share. 
# It prits some stats and qualitative results

new_data_x_file = "../datasets/Gen_data_Z_dim-200_mb_size-100_h_dim-600_preproc-1_beta-1.01_final_ly-Sigmoid_loss-BCELoss/new_x.npy"
new_data_y_file = "../datasets/Gen_data_Z_dim-200_mb_size-100_h_dim-600_preproc-1_beta-1.01_final_ly-Sigmoid_loss-BCELoss/new_y.npy"
actual_data_x_file = "../datasets/Eurlex/eurlex_docs/x_tr.npy"
actual_data_y_file = "../datasets/Eurlex/eurlex_docs/y_tr.npy"
indx2word_file = "../datasets/Eurlex/eurlex_docs/feature_names.txt"
indx2label = "../datasets/Eurlex/eurlex_docs/label_set.txt"
K = 10
# ----------------------------------------------------------------------------

new_data_x = np.load(new_data_x_file)
new_data_y = np.load(new_data_y_file)
actual_data_x = np.load(actual_data_x_file)
actual_data_y = np.load(actual_data_y_file)
f = open(indx2label, 'r')
temp = f.read().splitlines()
labels = []
for i in temp:
    labels.append(i.split(":")[1])
f = open(indx2word_file, 'r')
temp = f.read().splitlines()
words = []
for i in temp:
    words.append(i.split(":")[1])

print("Shapes: new_x: {}; new_y: {}; original_x: {}; original_y: {};".format(new_data_x.shape, \
                                    new_data_y.shape, actual_data_x.shape, actual_data_y.shape))
print("Num Words: {}; Num Labels: {};".format(len(labels), len(words)))

for data_pt_num in range(K):
    data_pt_labels = np.argwhere(new_data_y[data_pt_num]==1)
    label_names = []
    for label in data_pt_labels.tolist():
        # print(label)
        label_names.append(labels[label[0]])
    print("Labels in the data point : {}".format(label_names))

    data_pt_words = np.argsort(new_data_x[data_pt_num])[-10:]
    word_names = []
    for word in data_pt_words.tolist():
        word_names.append(words[word])
    print("Top 10 words in the data point : {}".format(word_names))

    # Nearest Data point in actual data
    indx = -1
    closest = 1e10
    # print(actual_data_y)
    for i in range(len(actual_data_y)):
        dist = -len(np.intersect1d(np.argwhere(actual_data_y[i]==1), np.argwhere(new_data_y[data_pt_num]==1)))
        # print(np.argwhere(actual_data_y[i]==1))
        # print(np.argwhere(new_data_y[data_pt_num]==1))
        if(dist<closest):
            closest = dist
            indx = i
    print(-closest)
    print(indx)
    data_pt_labels = np.argwhere(actual_data_y[indx]==1)
    label_names = []
    for label in data_pt_labels.tolist():
        label_names.append(labels[label[0]])
    print("Closest Label Set in the original data set has labels: {}".format(label_names))

    data_pt_words = np.argsort(actual_data_x[indx])[-10:]
    word_names = []
    for word in data_pt_words.tolist():
        word_names.append(words[word])
    print("Top 10 words in the data point with the above label: {}".format(word_names))

    print("="*50)