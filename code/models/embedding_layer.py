from header import *

class embedding_layer(torch.nn.Module):

    def __init__(self, params, embedding_weights):
        super(embedding_layer, self).__init__()
        self.l = nn.Embedding(params.vocab_size, params.embedding_dim)
        if params.model_variation == 'pretrain':
            self.l.weight.data.copy_(torch.from_numpy(embedding_weights))
            self.l.weight.requires_grad=False

    def forward(self, inputs):
        o = self.l(inputs)
        return o
