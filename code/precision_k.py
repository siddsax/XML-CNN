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
        num = np.sum(mat,axis=1)
        p[k] = np.mean(num/(k+1))

    return np.around(p, decimals=4)
