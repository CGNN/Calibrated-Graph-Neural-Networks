# Calibrated-Graph-Neural-Networks
Source code and data for "Learning Calibrated Graph Neural Network for Open-set Classification"

## Requirements
DGL 0.6

Pytorch

## Dataset
We provide Cora, CiterSeer and ogbn-arxiv(code to download) in this repository. DBLP dataset will be compressed and uploaded later (current size exceeds the Github limits)

## Open-set Classification 

### Run CGNN
We provide the default setting of CGNN, i.e. various GNN archs for classification GNN and DMoN for clustering GNN. Here, the n-epochs meaning the rounds of variational EM instead of training epochs for each module. We are improving the code to make the training epochs as a parameter as well.
```
python cgnn_dmon.py --dataset=cora --num-unseen=3 --n-epochs=10 --n-hidden=128 --dropout=0.0 --gnn-arch=gcn
```
In Table 7, we provide the generalizability of CGNN by varing gnn-archs and clustering algorithms. The DGI + DEC version is,
```
python cgnn_dgi.py --dataset=cora --n-epochs=200  --dropout=0.0 --self-train --cluster-gnn --num-unseen=3 --gnn-arch=gcn --n-hidden=128
```
One can change --gnn-arch into gcn, graphsage, gat, sgc as mentioned in the paper.

### Run Baseline
For example, we run GCN-DOC on Cora in the same setting with the paper (3 missing classes)
```
python semi_gcn_doc.py --dataset=cora --num-unseen=3 --n-epochs=200 --n-hidden=128 --dropout=0.0 --gnn-arch=gcn
```

All other baselines are available as semi_gcn_ths.py, semi_dgi_ths.py, semi_dgi_doc.py. Specifically, we run Feat.-X by setting --gnn-arch=mlp.
