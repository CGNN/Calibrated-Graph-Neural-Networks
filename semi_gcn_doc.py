import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
from DGI.models import DGI

from IPython import embed
from dgl.data import PPIDataset

import math
import networkx as nx

import utils
import argparse, pickle
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

from dgl.nn.pytorch.conv import SAGEConv,GraphConv,GATConv
import collections
from collections import Counter
from copy import deepcopy
from baselines.DMGI.utils import process
def ece_score(py, y_test, n_bins=20):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        #if a < 0.5:
        #    continue
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    #print(Bm)
    return ece / sum(Bm)

def cluster_accuracy(y_true, y_predicted, cluster_number = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.
    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy

class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h
    
    def output(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        self.activation = activation
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes)) # activation None

    def forward(self, features):
        h = features
        for idx,layer in enumerate(self.layers):
            if idx < len(self.layers) - 1:
                h = self.activation(layer(h))
            else:
                h = layer(h)
        return h
    
    def output(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

# class 
class Net(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=None)) # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h
    
    def output(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(GATConv(in_feats, n_hidden, num_heads=8, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden*8, n_hidden, num_heads=8, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(GATConv(n_hidden*8, n_classes, num_heads=1, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features):
        h = features
        #embed()
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](self.g, h).flatten(1)
        return self.layers[-1](self.g, h).mean(1)
    
    def output(self, g, features):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)
    

def dgi(args, new_classes):
    # training params
    batch_size = 1
    nb_epochs = 10000
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 128
    sparse = True
    # unk = True, if we have unseen classes
    unk = True
    nonlinearity = 'prelu' # special name to separate parameters

    # new_classes = [2]
    if args.dataset in ['cora', 'citeseer']:
        adj, features, one_hot_labels, idx_train, idx_val, idx_test = utils.load_data(args.dataset)
        idx_train, idx_val, in_idx_test, idx_test, out_idx_test, labels = utils.createDBLPTraining(one_hot_labels, idx_train, idx_val, idx_test, new_classes=new_classes)
        features = torch.FloatTensor(utils.preprocess_features(features))
    elif args.dataset == 'dblp':
        rownetworks, truefeatures_list, one_hot_labels, idx_train, idx_val, idx_test = process.load_data_dblp(args)
        # uncomment below line 
        #uncertain_idx = pickle.load(open('data/dblp_uncertain_2.p', 'rb'))
        #train_mask, val_mask, test_mask, uncertain_mask, labels = utils.createDBLPTraining(one_hot_labels, idx_train, idx_val, idx_test, new_classes=new_classes, idx_uncertain=idx_uncertain)
        labels = torch.LongTensor([np.where(r==1)[0][0] for r in one_hot_labels])
        
        #idx_train, labels, new_labels = utils.createClusteringData(one_hot_labels, idx_train, idx_val, idx_test, new_classes=new_classes)
        idx_train, idx_val, in_idx_test, idx_test, out_idx_test, labels, cluster_labels = utils.createOGBTraining(labels, idx_train, idx_val, idx_test, max_train=20, new_classes=new_classes) 
        # print(Counter(np.asarray(labels)[uncertain_idx]))
        features = torch.FloatTensor(utils.preprocess_features(truefeatures_list[0]))
    '''    
    elif args.dataset == 'dblp':
        rownetworks, truefeatures_list, one_hot_labels, idx_train, idx_val, idx_test = pickle.load(open('data/processed_dblp.p', 'rb'))
        idx_train, labels, new_labels = utils.createClusteringData(one_hot_labels, idx_train, idx_val, idx_test, new_classes=new_classes)
        in_idx_test = idx_test
        out_idx_test = list(set(range( len(labels))) - set(idx_test) - set(idx_train) - set(idx_val))
        idx_test = np.concatenate((idx_test, out_idx_test) , axis=0) 
        features = torch.FloatTensor(utils.preprocess_features(truefeatures_list[0]))
        one_hot_labels = torch.LongTensor(one_hot_labels)
    '''


    torch.cuda.set_device(args.gpu)

    if args.dataset == 'dblp':
        g =  DGLGraph(nx.Graph(rownetworks[0]))
        adj = g.adjacency_matrix()
    elif args.dataset in ['cora', 'citeseer']:
        # important to add self-loop
        # adj = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
        g =  DGLGraph(nx.Graph(adj+ sp.eye(adj.shape[0])))
    elif args.dataset == 'ogbn-arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name = args.dataset)
        g, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()
        labels = labels.reshape(-1)
        features = g.ndata['feat']
        
        #feat_smooth_matrix = calc_feat_smooth(adj, feat)
        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        major_labels = set()
        label_dist = Counter(labels[idx_train].tolist())
        for k in label_dist:
            if label_dist[k] > len(idx_train) / 40:
                major_labels.add(k)
        new_idx_train, new_idx_val, new_idx_test = [], [], []
        for idx in idx_train:
            if labels[idx].item() in major_labels:
                new_idx_train.append(idx.item())
        for idx in idx_val:
            if labels[idx].item() in major_labels:
                new_idx_val.append(idx.item())
        for idx in idx_test:
            if labels[idx].item() in major_labels:
                new_idx_test.append(idx.item())
        #idx_train = torch.LongTensor(new_idx_train)
        #idx_val = torch.LongTensor(new_idx_val)
        #idx_test = torch.LongTensor(new_idx_test)
        idx_train, idx_val, in_idx_test, idx_test, out_idx_test, labels, cluster_labels = utils.createOGBTraining(labels, new_idx_train, new_idx_val, new_idx_test, max_train=1000, new_classes=new_classes) 


    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    #if args.dataset == 'dblp':
    if False:
        nb_classes = max(labels[idx_val]).item() + 1
    else:
        if len(new_classes) > 0:
            nb_classes = max(labels[idx_val]).item()
        else:
            nb_classes = max(labels[idx_val]).item() + 1
    #embed()
    new_labels = deepcopy(labels)
    if args.dataset == 'ogbn-arxiv' or args.dataset == 'dblp':
        doc_labels = torch.zeros(features.shape[0], nb_classes + 2)
        labels[labels == -1] = nb_classes + 1
        doc_labels.scatter_(1, labels.view(labels.shape[0],1), 1)
    else:
        doc_labels = torch.zeros(features.shape[0], nb_classes + 1)
        doc_labels.scatter_(1, labels.view(labels.shape[0],1), 1)
    labels = doc_labels[:, :nb_classes]
    #embed()
    xent = nn.BCEWithLogitsLoss()
    print('number of classes {}'.format(nb_classes))

    if torch.cuda.is_available():
        print('Using CUDA')
        
        features = features.cuda()
        labels = labels.cuda()
        new_labels = new_labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        if args.dataset == 'ppi':
            test_labels = test_labels.cuda()
            val_labels = val_labels.cuda()
            val_features = val_features.cuda()
            test_features = test_features.cuda()
            # test_g = test_g.cuda()
            # val_g = val_g.cuda()
    train_lbls = labels[idx_train]

    # inductive setting
    if args.dataset == 'ppi':
        val_lbls = val_labels[idx_val]
        test_lbls = test_labels[idx_test]
    else:
        val_lbls = doc_labels[idx_val]
        test_lbls = new_labels[idx_test]
    
    
    cnt_wait = 0
    finetune = False
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    def weight_reset(m):
        if isinstance(m, GraphConv):
            m.reset_parameters()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.gnn_arch == 'graphsage':
        model = GraphSAGE(g.to(device),
                ft_size,
                args.n_hidden,
                nb_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                args.aggregator_type
                )
    elif args.gnn_arch == 'gat':
        model = GAT(g.to(device),
                ft_size,
                args.n_hidden,
                nb_classes,
                args.n_layers,
                F.elu,
                args.dropout,
                args.aggregator_type
                )
    elif args.gnn_arch == 'gcn':
        model = Net(g.to(device),
                ft_size,
                args.n_hidden,
                nb_classes,
                args.n_layers,
                F.elu,
                args.dropout,
                args.aggregator_type
                )
    elif args.gnn_arch == 'mlp':
        model = MLP(
                ft_size,
                args.n_hidden,
                nb_classes,
                args.n_layers,
                F.elu,
                args.dropout,
                args.aggregator_type
                )
    model.cuda()
    torch.manual_seed(0)
    #pre-train stage
    for _ in range(args.n_repeats):
        if args.dataset == 'dblp':
            unseen_val = torch.LongTensor(np.random.choice(list(out_idx_test), 150)).cuda()

        model.apply(weight_reset)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        
        for epoch in range(args.n_epochs):
            model.train()
            optimiser.zero_grad()
            logits = model(features)
            loss = xent(logits[idx_train], train_lbls) # + 0.25 * kl_div_fcn(F.log_softmax(logits[out_of_samples], dim=1), uniform_dist)
            preds = torch.argmax(logits[idx_val], dim=1)
        
            loss.backward()
            optimiser.step()
            #if epoch % 10 == 0:
            #    print(loss.item())
        
        #print(val_acc)
        model.eval()
        if args.dataset == 'ppi':
            embeds = model.output(val_g, val_features).detach()
        else:
            embeds = model(features).detach()
        
        if args.dataset == 'dblp' and False:
            val_embs = embeds[torch.cat([idx_val, unseen_val])]
            new_val_lbls = torch.cat([val_lbls, labels[unseen_val] ])
        else:
            val_embs = embeds[idx_val]
            new_val_lbls = val_lbls

        best_ths = []
        best_val_acc = -1
        if True:
            #ths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            #ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            if not unk:
                ths = [0.5]
            for _class in range(nb_classes):
            #for t in ths:
                #log.eval()
                #log_acc = []
                idx = new_labels[idx_val] == _class
                logits = torch.sigmoid(embeds[idx_val, :][idx, _class])
                pseudo_logits = 2-logits
                g_samples = logits.cpu().numpy().tolist() + pseudo_logits.cpu().numpy().tolist()
                std = math.sqrt(sum(map(lambda x: (x-1.0) ** 2, g_samples)) / len(g_samples))
                best_ths.append(max(0.5, 1 - 3*std))
        
        preds = torch.argmax(embeds[idx_test], dim=1)
        #print((torch.sum(preds == labels[in_idx_test].argmax(dim=1)).float() / labels[in_idx_test].shape[0]).item())
        
        #print(best_ths)
        logits = torch.sigmoid(embeds[idx_test])
        for idx, ti in enumerate(best_ths):
            logits[:, idx] = logits[:, idx] > ti
        unseen_pred = logits.sum(dim=1)==0
        #embed()
        if unk:
            preds[unseen_pred] = nb_classes
        # embed()
        micro_f1.append(f1_score(test_lbls.cpu(), preds.cpu(), average='micro'))
        macro_f1.append(f1_score(test_lbls.cpu(), preds.cpu(), average='macro'))
        # mirco_f1.append((torch.sum(preds == test_lbls).float() / test_lbls.shape[0]).item())
        preds = torch.argmax(embeds[in_idx_test], dim=1)
        # embed()
        in_acc.append((torch.sum(preds == new_labels[in_idx_test]).float() / new_labels[in_idx_test].shape[0]).item())
        
        if unk:
            preds_test = torch.zeros((idx_test.shape[0], nb_classes + 1))
            #embed()
            preds_test[:, :nb_classes] = torch.sigmoid(embeds[idx_test, :])
            #preds_test = F.softmax(embeds[idx_test], dim=1)
            # preds = torch.argmax(embeds[out_idx_test], dim=1)
            # preds[F.softmax(embeds[out_idx_test], dim=1).max(dim=1)[0].lt(best_ths)] = nb_classes
            # out_acc.append((torch.sum(preds == labels[out_idx_test]).float() / labels[out_idx_test].shape[0]).item())
            #embed()
            #unseen_pred = preds_test.max(dim=1)[0].lt(best_ths)
            if unseen_pred.sum() > 0:
                #pass
                preds_test[unseen_pred, -1] = 1 - preds_test[unseen_pred, :].max(dim=1)[0]
            preds_test = F.normalize(preds_test, p=1)
            #embed()
            out_acc.append(ece_score(preds_test.cpu().numpy(), labels[idx_test].cpu().numpy()))
        #embed() 
        #del embeds
        #torch.cuda.empty_cache()
    return in_acc, out_acc, micro_f1, macro_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--gnn-arch", type=str, default='gcn',
                        help="gnn arch of gcn/gat/graphsage")
    # parameter for PPI is 1000, 200 for Cora
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-out", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="print verbose step-wise information")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help=".")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--dataset',type=str, default='cora')
    parser.add_argument('--num-unseen',type=int, default=1)
    parser.add_argument('--model-name',type=str, default='graphsage')
    parser.add_argument('--metapaths', type=list, default=['PAP'])
    parser.add_argument('--new-classes', type=list, default=[])
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    args = parser.parse_args()
    if args.dataset == 'cora':
        num_class = 7
    elif args.dataset == 'ppi':
        num_class = 9
    elif args.dataset == 'citeseer':
        num_class = 6
    elif args.dataset == 'dblp':
        num_class = 5
    elif args.dataset == 'ogbn-arxiv':
        num_class = [2, 4, 5, 8, 10, 16, 24, 26, 28, 30, 34, 36]
    print(args)
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    
    for i in utils.generateUnseen(num_class, args.num_unseen):
        print('missing:', i)
        #if args.dataset == 'dblp':
        #    i = []
        _ , __, ___, ____ = dgi(args, i)
        in_acc += _
        out_acc += __
        micro_f1 += ___
        macro_f1 += ____
        #if args.dataset == 'dblp':
        #    break
        #break
        
    print(np.mean(in_acc), np.std(in_acc), np.mean(micro_f1), np.std(micro_f1))
    print(np.mean(macro_f1), np.std(macro_f1), np.mean(out_acc), np.std(out_acc))