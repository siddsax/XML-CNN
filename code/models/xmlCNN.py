from header import *
from cnn_encoder import cnn_encoder

class xmlCNN(nn.Module):
    def __init__(self, params, embedding_weights):
        super(xmlCNN, self).__init__()
        self.params = params
        self.embedding_layer = embedding_layer(params, embedding_weights)
        self.classifier = cnn_encoder(params)
        
    def forward(self, batch_x, batch_y):
        # ----------- Encode (X, Y) --------------------------------------------
        e_emb = self.embedding_layer.forward(batch_x)
        Y = self.classifier.forward(e_emb)
        loss = self.params.loss_fn(Y, batch_y)
        
        if(loss<0):
            print(cross_entropy)
            print(Y[0:100])
            print(batch_y[0:100])
            sys.exit()

        return loss.view(-1,1), Y