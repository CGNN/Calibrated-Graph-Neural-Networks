import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GNN, GraphSAGE, GAT, SGC

from dgl import DGLGraph
import dgl
from DGI.models import PPIDGI, DGI, LogReg
from baselines.DMGI.utils import process

from IPython import embed
#from dgl.data.citation_graph import CoraDataset
from sklearn.metrics.cluster import normalized_mutual_info_score

import networkx as nx
from sklearn import preprocessing
import math
import utils
import argparse, pickle
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from collections import Counter
from sklearn.metrics import f1_score
from ogb.nodeproppred import Evaluator

#from spherecluster import SphericalKMeans
#from subspace_clustering.cluster.selfrepresentation import ElasticNetSubspaceClustering
from scipy.optimize import linear_sum_assignment

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import sys
from dgl.nn import GraphConv
from dgl.data import PPIDataset
from copy import deepcopy

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
    #embed()
    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def match_accuracy(y_true, y_predicted, cost_matrix, nb_clusters):
    cluster_number = nb_clusters
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=1).reshape(-1, 1), "y_true": labels.reshape(-1, 1)})["acc"]

class DMoN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(DMoN, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, norm='both', weight=True, bias=True, activation=activation))
        self.n_classes = n_classes
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, norm='both', weight=True, bias=True, activation=activation))
        
        self.linear = nn.Linear(n_hidden, n_classes, bias=True)
        self.kl_div_fcn = torch.nn.KLDivLoss(reduction='mean')
        torch.nn.init.orthogonal_(self.linear.weight, gain=1)
        #torch.nn.init.kaiming_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, features, sp_adj, calibrate=None, M=None, calib_idx=None):
        num_edges, num_nodes = self.g.number_of_edges(), self.g.number_of_nodes()
        degrees = self.g.in_degrees() 
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        soft_assignment = F.softmax(self.linear(self.dropout(h)), dim=1)
        #embed()
        cluster_sizes = soft_assignment.sum(dim=0)
        #assignments_pooling = soft_assignment / cluster_sizes
        graph_pooled = torch.spmm(sp_adj, soft_assignment)
        
        graph_pooled = torch.matmul(soft_assignment.T, graph_pooled)
        normalized_c = torch.matmul(degrees.float(), soft_assignment).reshape(1, -1)
        normalizer = torch.matmul(normalized_c.T, normalized_c) / 2 / num_edges
        spectral_loss = -torch.trace(graph_pooled - normalizer) / 2 / num_edges
        collapse_loss = torch.norm(cluster_sizes) / num_nodes * math.sqrt(self.n_classes) - 1
        if calibrate is None:
            loss = spectral_loss + collapse_loss
            return loss, soft_assignment
        else:
            probs = F.normalize(soft_assignment+1e-3, p=1, dim=1)
            #probs = soft_assignment
            calibration_probs = torch.zeros_like(probs)
            calibration_probs[:, :calibrate.shape[1]] = calibrate
            
            calibration_loss = 0.5 *  self.kl_div_fcn(probs[calib_idx, :].matmul(M).log(), calibration_probs[calib_idx, :].detach())
            #loss = spectral_loss + collapse_loss + calibration_loss
            loss = calibration_loss + spectral_loss
            return loss, calibration_loss, soft_assignment
        

def train_classifier(model_p, optimizer_p, epochs, q_probs = None, in_domain_idx = None):
    xent = nn.CrossEntropyLoss()
    
    if q_probs is not None:
        q_samples = []
        if in_domain_idx is not None:
            for idx in in_domain_idx.tolist():
                # whether it's because mismatch
                #if labels[idx] == 4:
                if q_probs[idx] > 0.8:
                    q_samples.append(idx)
        else:
            for idx in range(features.shape[0]):
                # whether it's because mismatch
                #if labels[idx] == 4:
                if q_probs[idx] > 0.8:
                    q_samples.append(idx)
        #print("size of q samples", len(q_samples))
        #print(Counter(labels[q_samples].cpu().numpy().tolist()))
        #embed()
    for epoch in range(epochs):
        model_p.train()
        optimizer_p.zero_grad()
        logits = model_p(features)
        #print(epoch)
        if q_probs is None:
            loss = xent(logits[idx_train], labels[idx_train])
        else:
            #remember to change the below lines
            #v_samples = np.random.choice(idx_test, 20).tolist()
            #v_samples = np.random.choice(q_samples, min(20, len(q_samples))).tolist()
            # baseline running
            v_samples = np.random.choice(in_domain_idx.tolist(), 20).tolist()
            
            v_samples = torch.LongTensor(v_samples)
            loss = xent(logits[torch.cat([idx_train, v_samples])], torch.cat([labels[idx_train], nb_classes *  torch.ones_like(v_samples).to(device)]))
        #loss = xent(logits[idx_val], val_lbls)

        loss.backward()
        optimizer_p.step()

def train_cluster(model_q, optimizer_q, epochs, v_args = None):
    model_q.train()
    for epoch in range(epochs):
        optimizer_q.zero_grad()
        if v_args is None:
            loss, assignment = model_q(features, sp_adj)
        else:
            p_probs, M, _idx = v_args
            loss, calib_loss, assignment = model_q(features, sp_adj, p_probs, M, np.random.choice(_idx, 1))
            #loss, calib_loss, assignment = model_q(features, sp_adj, p_probs, M, _idx)
        loss.backward()
        optimizer_q.step()

        if epoch % 400 == 0 and False:
            reassignment, acc = cluster_accuracy(labels[idx_test].cpu().numpy(), assignment[idx_test].argmax(dim=1).cpu().numpy())
            print("all classes:", loss.item(), acc, normalized_mutual_info_score(labels[idx_test].cpu().numpy(), assignment[idx_test].argmax(dim=1).cpu().numpy()))

    if v_args is not None and False:
        #if False:
        print( (loss-calib_loss).item(), calib_loss.item())
        #reassignment, acc = cluster_accuracy(labels[idx_test].cpu().numpy(), assignment[idx_test].argmax(dim=1).cpu().numpy())
        #print("all classes:", acc, normalized_mutual_info_score(labels[idx_test].cpu().numpy(), assignment[idx_test].argmax(dim=1).cpu().numpy()))
        reassignment_1, acc_1 = cluster_accuracy(cluster_labels[~unseen_mask.numpy()], assignment[~unseen_mask].argmax(dim=1).cpu().numpy())
        print("seen classes:", acc_1, normalized_mutual_info_score(cluster_labels[~unseen_mask.numpy()], assignment[~unseen_mask].argmax(dim=1).cpu().numpy()))
        #print("seen classes:", acc_1, normalized_mutual_info_score(labels.cpu().numpy(), assignment.argmax(dim=1).cpu().numpy()))
        reassignment_2, acc_2 = cluster_accuracy(cluster_labels[unseen_mask.numpy()], assignment[unseen_mask].argmax(dim=1).cpu().numpy())
        print("unseen classes:", acc_2, normalized_mutual_info_score(cluster_labels[unseen_mask], assignment[unseen_mask].argmax(dim=1).cpu().numpy()))
        #embed()


def run(args, new_classes):
    global device, nb_classes, idx_train, idx_val, idx_test, in_idx_test, features, cluster_labels, labels, sp_adj, unseen_mask
    # new_classes = [2]
    #new_classes = []
    if args.dataset in  ['cora', 'citeseer']:
        # data = CoraDataset()
        
        adj, features, one_hot_labels, idx_train, idx_val, idx_test = utils.load_data(args.dataset)
        #idx_train = range(1000)
        #idx_val = range(1001, 1700)
        #idx_train, labels, new_labels = utils.createClusteringData(one_hot_labels, idx_train, idx_val, idx_test, new_classes=new_classes, unknown=unk)
        idx_train, idx_val, in_idx_test, idx_test, out_idx_test, labels = utils.createDBLPTraining(one_hot_labels, idx_train, idx_val, idx_test, max_train=20, new_classes=new_classes)
        
        #embed()
        features = torch.FloatTensor(utils.preprocess_features(features))
        # do I actually change the orders?
        cluster_labels = one_hot_labels.argmax(1)
    elif args.dataset == 'ogbn-arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name = args.dataset)
        g, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()
        labels = labels.reshape(-1)
        features = g.ndata['feat']
        adj = g.adjacency_matrix_scipy()
        
        #feat_smooth_matrix = calc_feat_smooth(adj, feat)
        evaluator = Evaluator(name=args.dataset)
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
        sp_adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
        #srcs, dsts = g.all_edges()
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
        cluster_labels = np.array(cluster_labels)
        in_domain_idx = (cluster_labels  != -1).nonzero()[0]
    
    torch.cuda.set_device(args.gpu)

    if args.dataset == 'dblp':
        #embed()
        g =  DGLGraph(nx.Graph(rownetworks[0]))
        adj = g.adjacency_matrix_scipy()
        #adj = utils.normalize_adj(rownetworks[0])
        sp_adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
        #print(type(sp_adj))
    elif args.dataset in ['cora', 'citeseer']:
        #g =  DGLGraph(nx.Graph(adj+ sp.eye(adj.shape[0])))
        g =  dgl.from_networkx(nx.Graph(adj+ sp.eye(adj.shape[0])))
        #g =  DGLGraph(nx.Graph(adj))
        #adj = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
        #sp_adj = g.adj()
        sp_adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    
    idx_train_set = set(idx_train)

    labels = torch.LongTensor(labels)
    #new_labels = torch.LongTensor(new_labels)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    #embed()
    if args.dataset == 'ppi':
        test_labels = test_labels.cuda()
        val_labels = val_labels.cuda()
        val_features = val_features.cuda()
        test_features = test_features.cuda()
        old_idx = torch.LongTensor(old_idx).cuda()
    
    if len(new_classes) > 0:
        nb_classes = max(labels[idx_val]).item()
    else:
        nb_classes = max(labels[idx_val]).item() + 1

    if len(new_classes) > 0:
        unseen_mask = (labels == nb_classes)
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    
    cnt_wait = 0
    best = 1e9
    best_t = 0
    #print('number of classes {}'.format(nb_classes))
    # uncomment below lines
    #nb_clusters = one_hot_labels.shape[1]
    
    '''
    opt = torch.optim.Adam([
                {'params': embedder.read.parameters()},
                {'params': embedder.disc.parameters()},
                {'params': embedder.gcn.parameters()},
                {'params': clusterer.parameters(), 'weight_decay': 1}
                #{'params': log.parameters(), 'lr': 1e-2},
                #{'params': relog.parameters(), 'lr': 1e-2}
            ], lr=lr, weight_decay=l2_coef)
    ''' 


    if torch.cuda.is_available():
        print('Using CUDA')        #clusterer = clusterer.cuda()
        features = features.cuda()
        labels = labels.cuda()
        #new_labels = new_labels.cuda()
        #idx_train = idx_train.cuda()
        #idx_val = idx_val.cuda()
        #idx_test = idx_test.cuda()
    
    # by default is sparse
    if sp_adj is not None:
        sp_adj = sp_adj.cuda()
    #else:
    #    adj = adj.cuda()

    
    train_lbls = labels[idx_train]
    # inductive setting
    if args.dataset == 'ppi':
        val_lbls = val_labels[idx_val]
        test_lbls = test_labels[idx_test]
    else:
        val_lbls = labels[idx_val]
        test_lbls = labels[idx_test]

    best_val_acc = 0
    cnt_wait = 0
    finetune = False
    in_acc, mirco_f1 = [], []
    uniform_dist = torch.ones(nb_classes).cuda() / nb_classes
    
    #feat_dgi = features.unsqueeze(0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_clusters = 7

    torch.manual_seed(0)
    if args.gnn_arch == 'gcn':
        model_p = GNN(g.to(device),
                ft_size,
                args.n_hidden,
                nb_classes + 1 if args.model_opt == 1 else nb_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                args.aggregator_type
                )
    elif args.gnn_arch == 'graphsage':
        model_p = GraphSAGE(g.to(device),
                ft_size,
                args.n_hidden,
                nb_classes + 1 if args.model_opt == 1 else nb_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                args.aggregator_type
                )
    elif args.gnn_arch == 'gat':
        model_p = GAT(g.to(device),
                ft_size,
                args.n_hidden,
                nb_classes + 1 if args.model_opt == 1 else nb_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                args.aggregator_type
                )
    elif args.gnn_arch == 'sgc':
        model_p = SGC(g.to(device),
                ft_size,
                args.n_hidden,
                nb_classes + 1 if args.model_opt == 1 else nb_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                args.aggregator_type
                )
    elif args.gnn_arch == 'gin':
        model_p = GIN(g.to(device),
            args.n_layers + 1,
            1,
            ft_size,
            args.n_hidden,
            nb_classes + 1,
            True,
            args.dropout,
            'sum',
            'sum')
    
    optimizer_p = torch.optim.Adam(model_p.parameters(), lr=0.01, weight_decay=args.weight_decay)
    model_p.cuda()

    # Pre-train classification GNN
    train_classifier(model_p, optimizer_p, 200)

    if False:
        #for epoch in range(1):
            #train_classifier(model_p, optimizer_p, 100)
        with torch.no_grad():
            model_p.eval()
            output = model_p(features)
            pred = F.softmax(model_p(features), dim=1)
            #embed()
            print(f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='micro'), f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='macro'))
    #embed()
    with torch.no_grad():
        model_p.eval()
        pred = F.softmax(model_p(features), dim=1)
        y = pred[idx_test].cpu().numpy()
        gt = labels[idx_test].cpu().numpy()
        #print(f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='micro'))
        #print(f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='macro'))
        
        #y[:, nb_classes].mean()
        #print(f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='micro'), f1_score(labels[in_idx_test].cpu(), pred[in_idx_test][:, :-1].argmax(dim=1).cpu(), average='micro'), ece_score(y, gt))
        #print("Full class F1: ", f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='micro'))
    
    model_q = DMoN(g.to(device),
            ft_size,
            args.n_hidden,
            nb_clusters,
            1,
            F.relu,
            args.dropout,
            args.aggregator_type
            )
    
    optimizer_q = torch.optim.Adam(model_q.parameters(), lr=args.lr)
    model_q.cuda()
    
    # comment below for disabling calibration
    train_cluster(model_q, optimizer_q, 2000)
    #embed()
    with torch.no_grad():
        model_p.eval()
        model_q.eval()
        pred_value = F.softmax(model_p(features), dim=1)
        pred_labels = torch.argmax(pred_value[in_idx_test], dim=1)
        loss, assignment = model_q(features, sp_adj)
        #pred_q = F.normalize(assignment[idx_test, :], p=1, dim=0)
        pred_q = torch.zeros_like(assignment[idx_test, :])
        #pred_q[:, assignment[idx_test, :].argmax(dim=1)] = 1
        idx = assignment[idx_test, :].argmax(dim=1).unsqueeze(1)
        pred_q.scatter_(1, idx, 1)
        pred_p = torch.zeros_like(pred_value[idx_test, :nb_classes])
        idx = pred_value[idx_test, :nb_classes].argmax(dim=1).unsqueeze(1)
        pred_p.scatter_(1, idx, 1)
        
        #pred_p = F.normalize(pred_p[idx_test, :nb_classes], p=1, dim=0)
        #distance_pq = 1000 * torch.ones((nb_clusters, nb_clusters))
        distance_pq = torch.zeros((nb_clusters, nb_clusters))
        distance_pq[:, :nb_classes] = pred_q.T.matmul(pred_p)
        distance_pq = distance_pq.max() - distance_pq
        #for idx in range(nb_clusters):
        #    
            #distance_pq[idx, :nb_classes] = (pred_p * (pred_p.log() - pred_q[:, idx].log().unsqueeze(1).repeat(1,nb_classes))).sum(dim=0)
        
        q_labels = np.argmax(assignment.cpu().numpy(), axis=1)
        #reassignment, accuracy = cluster_accuracy(pred_labels.cpu().numpy(), cluster_labels[in_idx_test])
        reassignment, accuracy = match_accuracy(pred_labels.cpu().numpy(), q_labels[in_idx_test], distance_pq.numpy(), nb_clusters)
        #reassignment, accuracy = cluster_accuracy(one_hot_labels, q_labels)
        #print("pre-train:", loss.item(), accuracy, normalized_mutual_info_score(labels[idx_test].cpu().numpy(), assignment[idx_test].argmax(dim=1).cpu().numpy()))
        #print(reassignment, accuracy)
        #print(reassignment, reassignment2)
        # same M as mentioned in Eq.10 and Eq.12
        M = torch.zeros((nb_clusters, nb_clusters)).cuda()
        #M = torch.zeros((nb_clusters, nb_classes)).cuda()
        unmatched = []
        unmatched_masks = np.zeros([nb_nodes], dtype=bool)
        matched_cluster = []
        for k,v in reassignment.items():
            M[k, v] = 1
            if v < nb_classes:
                
                matched_cluster.append(k)
            else:
                unmatched.append(k)
                unmatched_masks += cluster_labels == k 
        q_probs = assignment[:, unmatched ].sum(dim=1)
    
    best_val_acc = 0.0
    for _epoch in range(args.n_epochs):
        
        with torch.no_grad():
            model_q.eval()
            loss, assignment = model_q(features, sp_adj)
            q_probs = assignment[:, unmatched ].sum(dim=1)
        #optimizer_p = torch.optim.Adam(model_p.parameters(), lr=0.05, weight_decay=args.weight_decay)
        train_classifier(model_p, optimizer_p, 30, q_probs, in_domain_idx)
        #train_classifier(model_p, optimizer_p, 30, q_probs)

        with torch.no_grad():
            model_p.eval()
            p_probs = F.softmax(model_p(features)[:, :nb_classes], dim=1)
        
        # comment to disable calibration
        train_cluster(model_q, optimizer_q, 500, [p_probs, M, in_domain_idx])
        #train_cluster(model_q, optimizer_q, 500, [p_probs, M, np.arange(g.number_of_nodes())])

        with torch.no_grad():
            model_p.eval()
            pred = F.softmax(model_p(features), dim=1)
            y = F.normalize(pred[idx_test], p=1).cpu().numpy()
            gt = labels[idx_test].cpu().numpy()
            #y[:, nb_classes].mean()
            val_acc = f1_score(labels[idx_val].cpu(), pred[idx_val].argmax(dim=1).cpu(), average='micro')
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                torch.save(model_p.state_dict(), 'dumps/{}_best_cgnn_1.pkl'.format(args.dataset) )
            #print(f1_score(labels[in_idx_test].cpu(), pred[in_idx_test][:, :-1].argmax(dim=1).cpu(), average='micro'), ece_score(y, gt))
            #print("Tuning epoch:{}, Full class TEST F1:{}, Full class VAL F1:{}, ECE score:{}".format(_epoch, f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='micro'), f1_score(labels[idx_val].cpu(), pred[idx_val].argmax(dim=1).cpu(), average='micro'), ece_score(y, gt)))
    model_p.load_state_dict(torch.load('dumps/{}_best_cgnn_1.pkl'.format(args.dataset)))
    with torch.no_grad():
        model_p.eval()
        pred = F.softmax(model_p(features), dim=1)
        y = F.normalize(pred[idx_test], p=1).cpu().numpy()
        gt = labels[idx_test].cpu().numpy()
        in_acc = f1_score(labels[in_idx_test].cpu(), pred[in_idx_test][:, :-1].argmax(dim=1).cpu(), average='micro')
        mirco_f1 =  f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='micro')
        if mirco_f1 < 0.75:
            print("bad cases", new_classes)
        macro_f1 =  f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='macro')
        ece = ece_score(y, gt)
        print("Full class TEST F1:{}, Full class VAL F1:{}, ECE score:{}".format(f1_score(labels[idx_test].cpu(), pred[idx_test].argmax(dim=1).cpu(), average='micro'), f1_score(labels[idx_val].cpu(), pred[idx_val].argmax(dim=1).cpu(), average='micro'), ece_score(y, gt)))
    # calibrate cluster training
    if False:
        for epoch in range(args.n_epochs):
            model.train()
            optimiser.zero_grad()

            loss, calib_loss, assignment = model(features, sp_adj, one_hot_target, M, idx_test)
            #if args.dataset == 'ppi':
            #    loss = xent(logits, train_lbls)
            #if True:
            #    loss = xent(logits[idx_train], train_lbls) # + 0.25 * kl_div_fcn(F.log_softmax(logits[out_of_samples], dim=1), uniform_dist)
            if epoch % 100 == 0:
                print("losses", (loss-calib_loss).item(), calib_loss.item())
                reassignment_1, acc_1 = cluster_accuracy(labels[unseen_mask].cpu().numpy(), assignment[unseen_mask].argmax(dim=1).cpu().numpy())
                print("seen classes:", acc_1, normalized_mutual_info_score(labels.cpu().numpy(), assignment.argmax(dim=1).cpu().numpy()))
                reassignment_2, acc_2 = cluster_accuracy(labels[~unseen_mask].cpu().numpy(), assignment[~unseen_mask].argmax(dim=1).cpu().numpy())
                print("unseen classes:", acc_2, normalized_mutual_info_score(labels[~unseen_mask].cpu().numpy(), assignment[~unseen_mask].argmax(dim=1).cpu().numpy()))
            loss.backward()
            optimiser.step()
    #embed()
    return in_acc, mirco_f1,macro_f1, ece

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--gnn-arch", type=str, default='gcn',
                        help="gnn arch of gcn/gat/graphsage")
    parser.add_argument("--n-epochs", type=int, default=1500,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-out", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--num-unseen", type=int, default=0,
                        help="number of unseen classes")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="print verbose step-wise information")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help=".")
    parser.add_argument("--model-opt", type=int, default=1,
                        help="whether it's open classification")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--dataset',type=str, default='cora')
    parser.add_argument('--metapaths', type=list, default=['PAP'])
    parser.add_argument('--new-classes', type=list, default=[])
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    args = parser.parse_args()

    if args.dataset == 'ppi':
        data, val_data, test_data = PPIDataset('train'), PPIDataset('valid'), PPIDataset('test') 

    if args.dataset == 'cora':
        num_class = 7
    elif args.dataset == 'citeseer':
        num_class = 6
    elif args.dataset == 'dblp':
        num_class = 5
    elif args.dataset == 'ppi':
        num_class = 9
    elif args.dataset == 'ogbn-arxiv':
        num_class = [2, 4, 5, 8, 10, 16, 24, 26, 28, 30, 34, 36]

    print(args)
    in_acc, mirco_f1 = [], []
    macro_f1, ece = [], []
    #for i in range(1):
    for i in utils.generateUnseen(num_class, args.num_unseen):
        #i = (2,3,5)
        print("missing:", i)
        a , b, c, d = run(args, i)
        in_acc.append(a)
        mirco_f1.append(b)
        macro_f1.append(c)
        ece.append(d)
        #break
    #embed()
    print(np.mean(in_acc), np.std(in_acc), np.mean(mirco_f1), np.std(mirco_f1))
    print(np.mean(macro_f1), np.std(macro_f1), np.mean(ece), np.std(ece))