import numpy as np
import scipy.io as sio
def precision_k(true_mat, score_mat,k):
    p = np.zeros((k,1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k+1)]] = 0
        score_mat = np.ceil(score_mat)
        kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        num = np.sum(mat,1)
        p[k] = np.mean(num/(k+1))
    return p

# prec 92.55 85.22 75.08 61.33 51.96
# nDCG 92.55 88.02 86.13 86.20 86.84

# [[0.92395   ]
#  [0.852225  ]
#  [0.75086667]
#  [0.6133375 ]
#  [0.51962   ]]
