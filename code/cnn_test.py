from header import *
from collections import OrderedDict
from sklearn.metrics import log_loss

# def pass(a, b, model, x_tr, Y, params):
#     # e_emb = model.embedding_layer.forward(x_tr[i:i+params.mb_size].view(params.mb_size, x_te.shape[1]))
#     # Y[i:i+params.mb_size,:] = model.classifier(e_emb).data
#     e_emb = model.embedding_layer.forward(x_tr[a:b].view(params.mb_size, x_tr.shape[1]))
#     Y[a:b,:] = model.classifier(e_emb).data

#     return Y

def test_class(x_te, y_te, params, model=None, x_tr=None, y_tr=None, embedding_weights=None, verbose=True, save=True ):

    
    if(model==None):
        if(embedding_weights is None):
            print("Error: Embedding weights needed!")
            exit()
        else:
            model = xmlCNN(params, embedding_weights)
            # state_dict = torch.load(params.load_model + "/model_best", map_location=lambda storage, loc: storage)
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:]
            #     new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)
            # del new_state_dict
            model = load_model(model, params.load_model)
	   
    if(torch.cuda.is_available()):
        params.dtype_f = torch.cuda.FloatTensor
        params.dtype_i = torch.cuda.LongTensor
        model = model.cuda()
    else:
        params.dtype_f = torch.FloatTensor
        params.dtype_i = torch.LongTensor

    if(x_tr is not None and y_tr is not None):
        x_tr, _ = load_batch_cnn(x_tr, y_tr, params, batch=False)
        Y = np.zeros(y_tr.shape)
        rem = x_tr.shape[0]%params.mb_size 
        for i in range(0, x_tr.shape[0] - rem, params.mb_size ):
            e_emb = model.embedding_layer.forward(x_tr[i:i+params.mb_size].view(params.mb_size, x_te.shape[1]))
            Y[i:i+params.mb_size,:] = model.classifier(e_emb).data
 	if(rem):
           e_emb = model.embedding_layer.forward(x_tr[-rem:].view(rem, x_te.shape[1]))
           Y[-rem:, :] = model.classifier(e_emb).data   
        loss = log_loss(y_tr, Y)
        prec = precision_k(y_tr.todense(), Y, 5)
        print('Test Loss; Precision Scores [1->5] {} {} {} {} {} Cross Entropy {};'.format(prec[0], prec[1], prec[2], prec[3], prec[4],loss))
    
    
    x_te, _ = load_batch_cnn(x_te, y_te, params, batch=False)
    Y2 = np.zeros(y_te.shape)
    rem = x_te.shape[0]%params.mb_size
    for i in range(0,x_te.shape[0] - rem,params.mb_size):
        e_emb = model.embedding_layer.forward(x_te[i:i+params.mb_size].view(params.mb_size, x_te.shape[1]))
        Y2[i:i+params.mb_size,:] = model.classifier(e_emb).data

    if(rem):
        e_emb = model.embedding_layer.forward(x_te[-rem:].view(rem, x_te.shape[1]))
        Y2[-rem:,:] = model.classifier(e_emb).data

    loss = log_loss(y_te, Y2) # Reverse of pytorch
    #print("A")
    prec = precision_k(y_te.todense(), Y2, 5) # Reverse of pytorch
    print('Test Loss; Precision Scores [1->5] {} {} {} {} {} Cross Entropy {};'.format(prec[0], prec[1], prec[2], prec[3], prec[4],loss))
    
    if(save):
        Y_probabs2 = sparse.csr_matrix(Y2)
        sio.savemat('/'.join(params.load_model.split('/')[-1]) + '/score_matrix.mat' , {'score_matrix': Y_probabs2})

    return prec[0], loss
