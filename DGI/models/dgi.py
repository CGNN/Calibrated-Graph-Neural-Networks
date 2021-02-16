import torch
import torch.nn as nn
from ..layers import PPIGCN, GCN, AvgReadout, Discriminator, SelfExpress
import torch.nn.functional as F
from IPython import embed

class DGI(nn.Module):
    def __init__(self, n_in, n_h, n_classes, n_nodes, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        self.u = nn.Parameter(torch.Tensor(n_classes, n_h), requires_grad = True)
        self.kl_div_fcn = torch.nn.KLDivLoss(reduction='sum')
        

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret
    
    def finetune(self, clusterer, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        recon_loss, h_z = clusterer(h_1.squeeze())
        # h_1 = h_z.unsqueeze(0)
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)
        
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret, recon_loss


    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        #h_1_c = clusterer.output(h_1.squeeze())

        return h_1.squeeze().detach(), c.detach()

    def output_q(self, seq, adj, sparse, labels=None):
        #u = self.uembeds
        embeds = self.gcn(seq, adj, sparse).squeeze()

        return F.normalize(1.0 / torch.pow(embeds - self.u[:, None], 2).sum(2).T + 1, p=1)
    
    def ssl_loss(self, seq, adj, sparse, labels=None):
        #u = self.uembeds
        embeds = self.gcn(seq, adj, sparse).squeeze()
        self.q_dist = torch.zeros((embeds.shape[0], self.u.shape[0])).cuda()
        for i in range(self.u.shape[0]):
            self.q_dist[:, i] = 1.0 / torch.pow(embeds - self.u[i], 2).sum(dim=1) + 1
        
        self.q_dist = F.normalize(self.q_dist, p=1)
        f_j = torch.sum(self.q_dist, dim=0)
        self.p_dist = torch.pow(self.q_dist,2)
        for i in range(self.u.shape[0]):
            self.p_dist[:, i] /= f_j[i]
        if labels is not None:
            self.p_dist[labels>=0, :] = 1e-4
            self.p_dist[labels>=0, labels[labels>=0]] = 1.0

        #        self.p_dist[i, :] = torch.FloatTensor([0.96, 0.1, 0.1, 0.1]).cuda()
        #    elif labels[i] == 1:
        #        self.p_dist[i, :] = torch.FloatTensor([0.1, 0.96, 0.1, 0.1]).cuda()

        # it is very very important
        self.p_dist = F.normalize(self.p_dist, p=1).detach()
        kl = self.p_dist * torch.log(self.p_dist/self.q_dist)
        #embed()
        return kl.sum()

class PPIDGI(nn.Module):
    def __init__(self, n_in, n_h, n_classes, n_nodes, activation):
        super(PPIDGI, self).__init__()
        self.gcn = PPIGCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        self.u = nn.Parameter(torch.Tensor(n_classes, n_h), requires_grad = True)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)
    
        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret
    
    def finetune(self, clusterer, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        recon_loss, h_z = clusterer(h_1.squeeze())
        # h_1 = h_z.unsqueeze(0)
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)
        
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret, recon_loss


    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        #h_1_c = clusterer.output(h_1.squeeze())

        return h_1.squeeze().detach(), c.detach()

    def ssl_loss(self, seq, adj, sparse, labels=None):
        #u = self.uembeds
        embeds = self.gcn(seq, adj, sparse).squeeze()
        self.q_dist = torch.zeros((embeds.shape[0], self.u.shape[0])).cuda()
        for i in range(self.u.shape[0]):
            self.q_dist[:, i] = 1.0 / torch.pow(embeds - self.u[i], 2).sum(dim=1) + 1
        
        self.q_dist = F.normalize(self.q_dist, p=1)
        f_j = torch.sum(self.q_dist, dim=0)
        self.p_dist = torch.pow(self.q_dist,2)
        for i in range(self.u.shape[0]):
            self.p_dist[:, i] /= f_j[i]
        if labels is not None:
            self.p_dist[labels>=0, :] = 1e-4
            self.p_dist[labels>=0, labels[labels>=0]] = 1.0

        #        self.p_dist[i, :] = torch.FloatTensor([0.96, 0.1, 0.1, 0.1]).cuda()
        #    elif labels[i] == 1:
        #        self.p_dist[i, :] = torch.FloatTensor([0.1, 0.96, 0.1, 0.1]).cuda()

        # it is very very important
        self.p_dist = F.normalize(self.p_dist, p=1).detach()
        kl = self.p_dist * torch.log(self.p_dist/self.q_dist)
        #embed()
        return kl.sum()
    
    # def calibration_loss():



