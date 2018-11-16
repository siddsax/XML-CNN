from header import *
from cnn_train import *
from cnn_test import *
import pdb

# ------------------------ Params -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--zd', dest='Z_dim', type=int, default=100, help='Latent layer dimension')
parser.add_argument('--mb', dest='mb_size', type=int, default=20, help='Size of minibatch, changing might result in latent layer variance overflow')
# parser.add_argument('--hd', dest='h_dim', type=int, default=600, help='hidden layer dimension')
parser.add_argument('--lr', dest='lr', type=int, default=1e-3, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--e', dest='num_epochs', type=int, default=100, help='step for displaying loss')

parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=10, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
parser.add_argument('--ds', dest='data_set', type=str, default="rcv", help='dataset name')

parser.add_argument('--pp', dest='pp_flg', type=int, default=0, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')
parser.add_argument('--loss', dest='loss_type', type=str, default="BCELoss", help='Loss')

parser.add_argument('--hidden_dims', type=int, default=512, help='hidden layer dimension')
parser.add_argument('--sequence_length',help='max sequence length of a document', type=int,default=500)
parser.add_argument('--embedding_dim', help='dimension of word embedding representation', type=int, default=300)
parser.add_argument('--model_variation', help='model variation: CNN-rand or CNN-pretrain', type=str, default='pretrain')
parser.add_argument('--pretrain_type', help='pretrain model: GoogleNews or glove', type=str, default='glove')
parser.add_argument('--vocab_size', help='size of vocabulary keeping the most frequent words', type=int, default=30000)
parser.add_argument('--drop_prob', help='Dropout probability', type=int, default=.3)
parser.add_argument('--load_data', help='Load Data or not', type=int, default=0)
parser.add_argument('--mg', dest='multi_gpu', type=int, default=0, help='1 for 2 gpus and 0 for normal')
parser.add_argument('--filter_sizes', help='number of filter sizes (could be a list of integer)', type=int, default=[2, 4, 8], nargs='+')
parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=32)
parser.add_argument('--pooling_units', help='number of pooling units in 1D pooling layer', type=int, default=32)
parser.add_argument('--pooling_type', help='max or average', type=str, default='max')
parser.add_argument('--model_type', help='glove or GoogleNews', type=str, default='glove')
parser.add_argument('--num_features', help='50, 100, 200, 300', type=int, default=300)
parser.add_argument('--dropouts', help='0 for not using, 1 for using', type=int, default=0)
parser.add_argument('--clip', help='gradient clipping', type=float, default=1000)
parser.add_argument('--dataset_gpu', help='load dataset in full to gpu', type=int, default=1)
parser.add_argument('--dp', dest='dataparallel', help='to train on multiple GPUs or not', type=int, default=0)


params = parser.parse_args()

if(len(params.model_name)==0):
    params.model_name = "Gen_data_CNN_Z_dim-{}_mb_size-{}_hidden_dims-{}_preproc-{}_loss-{}_sequence_length-{}_embedding_dim-{}_params.vocab_size={}".format(params.Z_dim, params.mb_size, params.hidden_dims, params.pp_flg, params.loss_type, params.sequence_length, params.embedding_dim, params.vocab_size)

print('Saving Model to: ' + params.model_name)

# ------------------ data ----------------------------------------------
params.data_path = '../datasets/' + params.data_set
x_tr, x_te, y_tr, y_te, params.vocabulary, params.vocabulary_inv, params = save_load_data(params, save=params.load_data)

params = update_params(params)
# -----------------------  Loss ------------------------------------
params.loss_fn = torch.nn.BCELoss(size_average=False)
# -------------------------- Params ---------------------------------------------
if params.model_variation=='pretrain':
    embedding_weights = load_word2vec(params)
else:
    embedding_weights = None

if torch.cuda.is_available():
    params.dtype = torch.cuda.FloatTensor
else:
    params.dtype = torch.FloatTensor


if(params.training):
    train(x_tr, y_tr, x_te, y_te, embedding_weights, params)

else:
    test_class(x_te, y_te, params, x_tr=x_tr, y_tr=y_tr, embedding_weights=embedding_weights)
