import sys
sys.path.append('utils/')
sys.path.append('models/')
import numpy as np
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import argparse
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cPickle
from sklearn.metrics import silhouette_score
from dpmeans import *
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import scipy.io as sio
x_tr = np.load('datasets/Eurlex/eurlex_docs/x_tr.npy')
y_tr = np.load('datasets/Eurlex/eurlex_docs/y_tr.npy')
x_te = np.load('datasets/Eurlex/eurlex_docs/x_te.npy')
y_te = np.load('datasets/Eurlex/eurlex_docs/y_te.npy')

n = np.shape(x_tr)[0]
m = np.shape(y_tr)[1]



# ------ Making Adjacency ------------------
dct = {}
for i in range(m):
    dct[i] = np.argwhere(y_tr[:,i]==1)

adjacency_mat = np.zeros((m,m))
check_mat = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        adjacency_mat[i,j] = len(np.intersect1d(dct[i],dct[j]))
        adjacency_mat[j, i] = adjacency_mat[i,j]
        check_mat[i,j] = check_mat[j,i] = 1
    # adjacency_mat[i, i] = len(dct[i])
    # check_mat[i,i] = 1
    
    print(i)
np.save('adjacency_mat', adjacency_mat)
adjacency_mat = sparse.csr_matrix(adjacency_mat)
sio.savemat('adjacency_mat', adjacency_mat)
print((check_mat==0).any())
print(adjacency_mat[:100,:100])
# -----------------------------------------

#  ------------- PP ---------------------------------------
adjacency_mat = np.load('/scratch/work/saxenas2/CVAE_XML/adjacency_mat.npy')
pp = preprocessing.MinMaxScaler()
scaler = pp.fit(adjacency_mat)
adjacency_mat = scaler.transform(adjacency_mat)
#  -------------------------------------------------------

#  ----------------------- cluster + score ---------------
clusters = [2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30]
scores = []
scores_silhoette = []
for cluster_no in clusters:
    print(cluster_no)
    kmeans = KMeans(n_clusters=cluster_no, random_state=0).fit(adjacency_mat)
    scores.append(kmeans.score(adjacency_mat))
    label = kmeans.labels_
    scores_silhoette.append(silhouette_score(adjacency_mat, label, metric='euclidean'))
    with open('classifier_' + str(cluster_no) + '.pkl', 'wb') as fid:
        cPickle.dump(kmeans, fid)
# ---------------------------------------------------------

# scores = []
# for cluster_no in clusters:
#     with open('classifier_'+ str(cluster_no) + '.pkl', 'rb') as fid:
#         kmeans = cPickle.load(fid)
#         label = kmeans.labels_
#         scores.append(silhouette_score(adjacency_mat, label, metric='euclidean'))

matplotlib.pyplot.plot(clusters, scores)
plt.show()

# ---------------------- Explore Clusters -------------------------
cluster_no = 30
# with open('clusterings/classifier_'+ str(cluster_no) + '.pkl', 'rb') as fid:
with open('classifier_'+ str(cluster_no) + '.pkl', 'rb') as fid:
    kmeans = cPickle.load(fid)

y_pred = kmeans.predict(adjacency_mat)
clusters = {}
y_of_cluster = {}
for i in range(cluster_no):
    clusters[i] = np.argwhere(y_pred==i)
    y_of_cluster[i] = y_tr[:, clusters[i]]
    # y_of_cluster[i] = np.array(y_of_cluster[i][:,0])
    x = np.sum(y_tr, 0)
    y = np.sum(y_of_cluster[i], 0)
    mean_labels = np.mean(np.sum(y_of_cluster[i], 0))
    top5_labels = np.argsort(y)[-10:]
    top5_label_counts = np.sort(y)[-10:]
    num_tail_labels_1 = len(np.argwhere(x[clusters[i]]<=1))
    num_tail_labels_2 = len(np.argwhere(x[clusters[i]]<=2))
    num_tail_labels_5 = len(np.argwhere(x[clusters[i]]<=5))   

    print("No. of Labels {6}; Mean No. of Labels {0}; top 5 labels {1}, top 5 label counts {2}; num tail labels(1) \
    {3}; num tail labels(2) {4}; num tail labels(5) {5}".format(mean_labels, top5_labels, top5_label_counts,  
    num_tail_labels_1, num_tail_labels_2, num_tail_labels_5, len(clusters[i])))
# ---------------------- Explore Clusters -------------------------
