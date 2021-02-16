import torch
import torch.nn as nn
from IPython import embed

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)


class PPIGCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(PPIGCN, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(in_ft, out_ft, bias=False),nn.Linear(out_ft, out_ft, bias=False),nn.Linear(out_ft, out_ft, bias=False)])
        self.skip =  nn.Linear(in_ft, out_ft, bias=False)
        # self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(len(self.fc), out_ft))
            #for m in self.bias:
            #    m.data.fill_(0.0)
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        for idx,layer in enumerate(self.fc):
            if idx == 0:
                seq_fts = layer(seq)
            elif idx == 1:
                # embed()
                seq_fts = layer(out + self.skip(seq))
                prev_out = out
            else:
                seq_fts = layer(out + prev_out +self.skip(seq))
        
        
            if sparse:
                out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            else:
                out = torch.bmm(adj, seq_fts)
            out = self.act(out+self.bias[idx, :])
        
        #if self.bias is not None:
        #    out += self.bias
        #seq_fts = self.fc_2(self.act(out))
        
        return out