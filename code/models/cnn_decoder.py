from header import *

class cnn_decoder(nn.Module):
    def __init__(self, params):
        super(cnn_decoder, self).__init__()
        self.params = params
        self.out_size = self.params.decoder_kernels[-1][0]
        
        self.bn_1 = nn.BatchNorm1d(self.params.sequence_length + 1)
        self.drp = nn.Dropout(p=params.drop_prob)
        self.conv_layers = nn.ModuleList()
        self.bn_x = nn.ModuleList()
        self.relu = nn.ReLU()
        for layer in range(len(params.decoder_kernels)):
            [out_chan, in_chan, width] = params.decoder_kernels[layer]
            layer = nn.Conv1d(in_chan, out_chan, width,
                         dilation=self.params.decoder_dilations[layer],
                         padding=self.params.decoder_paddings[layer])
            torch.nn.init.xavier_uniform_(layer.weight)
            bn_layer = nn.BatchNorm1d(out_chan)
            self.conv_layers.append(layer)
            self.bn_x.append(bn_layer)
        
        # self.bn_2 = nn.BatchNorm1d(self.out_size)
        self.fc = nn.Linear(self.out_size, self.params.vocab_size)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, decoder_input, z, batch_y):
        [batch_size, seq_len, embed_size] = decoder_input.size()
        z = torch.cat([z, batch_y], 1)
        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.Z_dim + self.params.H_dim)
        x = torch.cat([decoder_input, z], 2)
        x = x.transpose(1, 2).contiguous()
        x = self.drp(x)
        for layer in range(len(self.params.decoder_kernels)):
            x = self.conv_layers[layer](x)
            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()
            x = self.relu(x)
            x = self.bn_x[layer](x)
        x = x.transpose(1, 2).contiguous()
        if(self.params.multi_gpu):
            x = x.cuda(2)
            x = self.fc(x)#.cuda(1)
        else:
            x = self.fc(x)
        x = x.view(-1, seq_len, self.params.vocab_size)
        return x
