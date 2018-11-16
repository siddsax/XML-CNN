from header import *
class classifier(nn.Module):
    def __init__(self, params):
        super(classifier, self).__init__()
        self.params = params
        if(self.params.dropouts):
            self.drp = nn.Dropout(.5)
        self.l1 = nn.Linear(params.h_dim, params.H_dim)
        self.l2 = nn.Linear(params.H_dim, params.y_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.l1.weight)

    def forward(self, H):
        H = self.l1(H)
        H = self.relu(H)
        H = self.l2(H)
        H = self.sigmoid(H)
        return H