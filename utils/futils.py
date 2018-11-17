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
import data_helpers
import scipy
import subprocess
from scipy import sparse
def weights_init(m):
    if(torch.__version__=='0.4.0'):
    	torch.nn.init.xavier_uniform_(m)
    else:
	torch.nn.init.xavier_uniform(m)
def get_gpu_memory_map(boom, name=False):
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    if(name):
        print("In " + str(name) + " Print: {0}; Mem(1): {1}; Mem(2): {2}; Mem(3): {3}; Mem(4): {4}".format( boom, gpu_memory_map[0], \
        gpu_memory_map[1], gpu_memory_map[2], gpu_memory_map[3]))
    else:
        print("Print: {0}; Mem(1): {1}; Mem(2): {2}; Mem(3): {3}; Mem(4): {4}".format( boom, gpu_memory_map[0], \
        gpu_memory_map[1], gpu_memory_map[2], gpu_memory_map[3]))
    return boom+1


def count_parameters(model):
    a = 0
    for p in model.parameters():
        if p.requires_grad:
            a += p.numel()
    return a

def effective_k(k, d):
    return (k - 1) * d + 1

def load_model(model, name, optimizer=None):
    if(torch.cuda.is_available()):
        checkpoint = torch.load(name)
    else:
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        init = checkpoint['epoch']
        return model, optimizer, init
    else:
        return model

def save_model(model, optimizer, epoch, name):
  
   
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, "../saved_models/" + name)
def sample_z(mu, log_var, params, dtype_f):
    eps = Variable(torch.randn(params.batch_size, params.Z_dim).type(dtype_f))
    k = torch.exp(log_var / 2) * eps
    return mu + k
def gen_model_file(params):
    data_name = params.data_path.split('/')[-2]
    fs_string = '-'.join([str(fs) for fs in params.filter_sizes])
    file_name = 'data-%s_sl-%d_ed-%d_fs-%s_nf-%d_pu-%d_pt-%s_hd-%d_bs-%d_model-%s_pretrain-%s_beta-%s' % \
        (data_name, params.sequence_length, params.embedding_dim,
         fs_string, params.num_filters, params.pooling_units,
         params.pooling_type, params.hidden_dims, params.batch_size,
         params.model_variation, params.pretrain_type, params.beta)
    return file_name

def sample_z(mu, log_var, params):
    eps = Variable(torch.randn(log_var.shape[0], params.Z_dim).type(params.dtype))
    k = torch.exp(log_var / 2) * eps
    return mu + k

def load_data(X, Y, params, batch=True):
    if(batch):
        a = np.random.randint(0,params.N, size=params.mb_size)
        if isinstance(X, scipy.sparse.csr.csr_matrix) or isinstance(X, scipy.sparse.csc.csc_matrix):
            X, c = X[a].todense(), Y[a].todense()
        else:
            X, c = X[a], Y[a]
            
    else:
        if isinstance(X, scipy.sparse.csr.csr_matrix) or isinstance(X, scipy.sparse.csc.csc_matrix):
            X, c = X.todense(), Y.todense()
        else:
            X, c = X, Y
    
    X = Variable(torch.from_numpy(X.astype('float32')).type(params.dtype))
    Y = Variable(torch.from_numpy(c.astype('float32')).type(params.dtype))
    return X,Y

def write_grads(model, thefile):
    grads = []
    for key, value in model.named_parameters():
        if(value.grad is not None):
            grads.append(value.grad.mean().squeeze().cpu().numpy())

    thefile = open('gradient_classifier.txt', 'a+')
    for item in grads:
        thefile.write("%s " % item)
    thefile.write("\n" % item)
    thefile.close()

def save_load_data(params, save=0):
    params.pad_token = "<PAD/>"
    params.go_token = '<GO/>'
    params.end_token = '<END/>'

    if(save):
        print("Loading Data")
        #####################################################
        params.data_path += '.p'
        x_tr, y_tr, x_te, y_te, vocabulary, vocabulary_inv, params = data_helpers.load_data(params, max_length=params.sequence_length, vocab_size=params.vocab_size)
        x_tr = x_tr.astype(np.int32)
        x_te = x_te.astype(np.int32)
        y_tr = y_tr.astype(np.int32)
        y_te = y_te.astype(np.int32)
        #####################################################
        params.data_path = params.data_path[:-2]
        if not os.path.exists(params.data_path):
            os.makedirs(params.data_path)
        
        x_tr = sparse.csr_matrix(x_tr)
        x_te = sparse.csr_matrix(x_te)
        sparse.save_npz(params.data_path + '/x_train', x_tr)
        sparse.save_npz(params.data_path + '/y_train', y_tr)
        sparse.save_npz(params.data_path + '/y_test', y_te)
        sparse.save_npz(params.data_path + '/x_test', x_te)
        np.save(params.data_path + '/vocab', vocabulary)
        np.save(params.data_path + '/vocab_inv', vocabulary_inv)

    x_tr = sparse.load_npz(params.data_path + '/x_train.npz')
    y_tr = sparse.load_npz(params.data_path + '/y_train.npz')
    x_te = sparse.load_npz(params.data_path + '/x_test.npz')
    y_te = sparse.load_npz(params.data_path + '/y_test.npz')

    vocabulary = np.load(params.data_path + '/vocab.npy').item()
    vocabulary_inv = np.load(params.data_path + '/vocab_inv.npy')
    params.X_dim = x_tr.shape[1]
    params.y_dim = y_tr.shape[1]
    params.N = x_tr.shape[0]
    params.vocab_size = len(vocabulary)
    params.classes = y_tr.shape[1]

    return x_tr, x_te, y_tr, y_te, vocabulary, vocabulary_inv, params

def load_batch_cnn(x_tr, y_tr, params, batch=True, batch_size=0, decoder_word_input=None, decoder_target=None, testing=0):

    indexes = 0 # for scope
    if(batch):
        if(batch_size):
            params.go_row = np.ones((batch_size,1))*params.vocabulary[params.go_token]
            params.end_row = np.ones((batch_size,1))*params.vocabulary[params.end_token]
            indexes = np.array(np.random.randint(x_tr.shape[0], size=batch_size))
            x_tr, y_tr = x_tr[indexes,:], y_tr[indexes,:]
        else:
            params.go_row = np.ones((params.mb_size,1))*params.vocabulary[params.go_token]
            params.end_row = np.ones((params.mb_size,1))*params.vocabulary[params.end_token]
            indexes = np.array(np.random.randint(x_tr.shape[0], size=params.mb_size))
            x_tr, y_tr = x_tr[indexes,:], y_tr[indexes,:]
    else:
        params.go_row = np.ones((x_tr.shape[0],1))*params.vocabulary[params.go_token]
        params.end_row = np.ones((x_tr.shape[0],1))*params.vocabulary[params.end_token]

    x_tr = x_tr.todense()
    y_tr = y_tr.todense()

    x_tr = Variable(torch.from_numpy(x_tr.astype('int')).type(params.dtype_i))
    if(testing==0):
        y_tr = Variable(torch.from_numpy(y_tr.astype('float')).type(params.dtype_f))

    return x_tr, y_tr
    
def update_params(params):
    if(len(params.model_name)==0):
        params.model_name = gen_model_file(params)
    params.decoder_kernels = [(400, params.Z_dim + params.hidden_dims + params.embedding_dim, 3),
                                (450, 400, 3),
                                (500, 450, 3)]
    params.decoder_dilations = [1, 2, 4]
    params.decoder_paddings = [effective_k(w, params.decoder_dilations[i]) - 1
                                    for i, (_, _, w) in enumerate(params.decoder_kernels)]

    return params


def sample_word_from_distribution(params, distribution):
    ix = np.random.choice(range(params.vocab_size), p=distribution.view(-1))
    x = np.zeros((params.vocab_size, 1))
    x[ix] = 1
    return params.vocabulary_inv[np.argmax(x)]
