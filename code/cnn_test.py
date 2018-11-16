from header import *
from collections import OrderedDict
from sklearn.metrics import log_loss

def test_class(x_te, y_te, params, model=None, x_tr=None, y_tr=None, embedding_weights=None, verbose=True, save=True ):

    
    if(model==None):
        if(embedding_weights is None):
            print("Error: Embedding weights needed!")
            exit()
        else:
            model = cnn_encoder_decoder(params, embedding_weights)
            # state_dict = torch.load(params.load_model + "/model_best", map_location=lambda storage, loc: storage)
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:]
            #     new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)
            # del new_state_dict
            model.load_state_dict(torch.load(params.load_model, map_location=lambda storage, loc: storage))
    if(torch.cuda.is_available()):
        params.dtype_f = torch.cuda.FloatTensor
        params.dtype_i = torch.cuda.LongTensor
        model = model.cuda()
    else:
        params.dtype_f = torch.FloatTensor
        params.dtype_i = torch.LongTensor

    if(x_tr is not None and y_tr is not None):
        x_tr, _, _, _ = load_batch_cnn(x_tr, y_tr, params, batch=False)
        Y = np.zeros(y_tr.shape)
        rem = x_tr.shape[0]%params.mb_size
        e_emb = model.embedding_layer.forward(x_tr[-rem:].view(rem, x_te.shape[1]))
        H = model.encoder.forward(e_emb)
        Y[-rem:, :] = model.classifier(H).data
        for i in range(0, x_tr.shape[0] - rem, params.mb_size ):
            print(i)
            e_emb = model.embedding_layer.forward(x_tr[i:i+params.mb_size].view(params.mb_size, x_te.shape[1]))
            H = model.encoder.forward(e_emb)
            Y[i:i+params.mb_size,:] = model.classifier(H).data
    
        loss = log_loss(y_tr, Y)
        prec = precision_k(y_tr, Y, 5)
        print('Test Loss; Precision Scores [1->5] {} Cross Entropy {};'.format(prec, loss))

    #y_te = y_te[:,:-1]
    x_te, _ = load_batch_cnn(x_te, y_te, params, batch=False)
    Y2 = np.zeros(y_te.shape)
    rem = x_te.shape[0]%params.mb_size
    for i in range(0,x_te.shape[0] - rem,params.mb_size):
        # print(i)
        e_emb = model.embedding_layer.forward(x_te[i:i+params.mb_size].view(params.mb_size, x_te.shape[1]))
        H2 = model.encoder.forward(e_emb)
        Y2[i:i+params.mb_size,:] = model.classifier(H2).data

    if(rem):
        e_emb = model.embedding_layer.forward(x_te[-rem:].view(rem, x_te.shape[1]))
        H2 = model.encoder.forward(e_emb)
        Y2[-rem:,:] = model.classifier(H2).data

    loss = log_loss(y_te, Y2) # Reverse of pytorch
    prec = precision_k(y_te, Y2, 5) # Reverse of pytorch
    print('Test Loss; Precision Scores [1->5] {} Cross Entropy {};'.format(prec, loss))

    if(save):
        Y_probabs2 = sparse.csr_matrix(Y2)
        sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs2})

    return prec[0], loss
