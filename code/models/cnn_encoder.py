from header import *

def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class cnn_encoder(torch.nn.Module):
    
    def __init__(self, params):
        super(cnn_encoder, self).__init__()
        self.params = params
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_x = nn.ModuleList()
        fin_l_out_size = 0
        
        self.bn_1 = nn.BatchNorm1d(params.embedding_dim)
        if(params.dropouts):
            self.drp = nn.Dropout(p=.25)

        for fsz in params.filter_sizes:
            l_out_size = out_size(params.sequence_length, fsz, stride=2)
            pool_size = l_out_size // params.pooling_units
            l_conv = nn.Conv1d(params.embedding_dim, params.num_filters, fsz, stride=2)
            torch.nn.init.xavier_uniform_(l_conv.weight)
            if params.pooling_type == 'average':
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                pool_out_size = (int((l_out_size - pool_size)/pool_size) + 1)*params.num_filters
            elif params.pooling_type == 'max':
                l_pool = nn.MaxPool1d(2, stride=1)
                pool_out_size = (int(l_out_size*params.num_filters - 2) + 1)
            fin_l_out_size += pool_out_size
            bn_x = nn.BatchNorm1d(pool_out_size)
            self.conv_layers.append(l_conv)
            self.bn_x.append(bn_x)
            self.pool_layers.append(l_pool)

        self.fin_layer = nn.Linear(fin_l_out_size, params.h_dim)
        self.bn_2 = nn.BatchNorm1d(params.H_dim)
        torch.nn.init.xavier_uniform_(self.fin_layer.weight)
        # self.mu = nn.Linear(fin_l_out_size, params.Z_dim, bias=True)
        # self.var = nn.Linear(fin_l_out_size, params.Z_dim, bias=True)
        # torch.nn.init.xavier_uniform_(self.var.weight)
        # torch.nn.init.xavier_uniform_(self.mu.weight)

    def forward(self, inputs):
        # o0 = self.drp(self.bn_1(inputs)).permute(0,2,1)
        o0 = inputs.permute(0,2,1)# self.bn_1(inputs.permute(0,2,1))
        if(self.params.dropouts):
            o0 = self.drp(o0) 
        conv_out = []

        for i in range(len(self.params.filter_sizes)):
            o = self.conv_layers[i](o0)
            o = o.view(o.shape[0], 1, o.shape[1]*o.shape[2])
            o = self.pool_layers[i](o)
            o = nn.functional.relu(o)
            o = o.view(o.shape[0],-1)
            # o = self.bn_x[i](o)
            conv_out.append(o)
            del o
        del o0
        if len(self.params.filter_sizes)>1:
            o = torch.cat(conv_out,1)
        else:
            o = conv_out[0]
        del conv_out
        o = self.fin_layer(o)
        o = nn.functional.relu(o)
        # o = self.bn_2(o)
        if(self.params.dropouts):
            o = self.drp(o) 
        return o