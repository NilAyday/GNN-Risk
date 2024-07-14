import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
import torch.optim as optim
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.model_selection
import collections

import os
import torch
#from torch_geometric.nn import GCNConv, GATConv, ChebConv,  JumpingKnowledge

#from torch_geometric.utils import add_self_loops, get_laplacian
from torch.nn import Sequential, Linear, ReLU,ModuleList
#from torch_geometric.nn.dense.linear import Linear as dense_Linear
import torch.nn.functional as F
import torch.nn as nn
#from torch_geometric.utils import add_self_loops, degree, remove_self_loops,to_undirected
from torch.nn import Linear, Parameter
import math

from model import *

def rdm_feature(X, percent):
    """Swap randomly rows in feature matrix with percentage p (p>0)."""

    X_l = X.tolist()
    n = len(X_l)

    X_indeX_l = list(range(0,n))

    sample_index = sorted(random.sample(X_indeX_l, int(n*percent/100)))
    # To actually copy the list
    sample_index_init = sample_index[:]

    random.shuffle(sample_index)
    # To actually copy the list
    sample_index_shuffled = sample_index[:]

    X_l_shuffled = []
    for i in range(0,n):
        if i not in sample_index_init:
            X_l_shuffled.append(X_l[i])
        else:
            related_index_shuffled = sample_index_shuffled[sample_index_init.index(i)]
            X_l_shuffled.append(X_l[related_index_shuffled])

    num_diff = 0
    for i,j in zip(X_l,X_l_shuffled):
        if (i != j):
            num_diff = num_diff + 1

    num_diff_rate = num_diff / len(X_l)

    X_l_shuffled = np.array(X_l_shuffled,dtype=np.float32)

    return torch.from_numpy(X_l_shuffled)

device='cuda'
adj, feature, labels,idx_train,idx_val,idx_test = load_citation('pubmed')


A=adj.to_dense()
num_classes=len(np.unique(labels))
G= nx.from_numpy_array(adj.cpu().detach().to_dense().numpy())

cudaid = "cuda"
device = torch.device(cudaid)
features = feature.to(device)
adj = adj.to(device)


feature_size=features.shape[1]
num_classes = len(torch.unique(labels))
n=features.shape[0]

epochs=100
patience=10
initial_features=features
percentages=[0.1,10,20,30,40,50,60,70,80,90,100]

def train(model,features,adj,optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate(model,features,adj,optimizer):
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test(model,features,adj,optimizer):
    #model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()


models =  ['AH','[A H]','AH (X=I)']
#['AH','[A H]','AH (A=I)','[0.8*AH 0.2*A]']# Replace with your actual model names or instances
percentages = [0.1,10,20,30,40,50,60,70,80,90,100]

# Dictionary to store test accuracies for each model
model_test_mean_acc = {model: [] for model in models}
model_test_std_acc = {model: [] for model in models}

for model_name in models:
    print(f"Evaluating {model_name}")
    if model_name== 'AH':
        hidden_dim=64
        dropout=0.3
        # Best combination: model_0_dataset=wikipedia2_nhid=[128]_dropout=0.4_epochs=400_lr=0.001_wd=0.005_patience=100_runs=10.pkl
        model=my_GCN(feature_size,[64],num_classes,dropout,my_GraphConvolution13,n)
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.005,weight_decay=0.005)
    elif model_name== '[A H]':
        dropout=0.2
        model=my_GCN(feature_size,[16],num_classes,dropout,my_GraphConvolution14,n)
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.005,weight_decay=0.001)
    elif model_name== 'AH (X=I)':
        dropout=0.5
        #Best combination: model_13_dataset=pubmed_nhid=[512, 128, 64]_dropout=0.5_epochs=400_lr=0.001_wd=0.0001_patience=100_runs=10.pkl         
        model=my_GCN_X_I(feature_size,[512, 128, 64],num_classes,dropout,my_GraphConvolution13,n)
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    elif model_name== '[0.8*AH 0.2*A]':
        dropout=0.5
        model=my_GCN(feature_size,[64],num_classes,dropout,my_GraphConvolution12,n)                                          
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=0.001)
    elif model_name== '[0.8*AH 0.1*H 0.1*X]':
        dropout=0.2
        model=my_GCN(feature_size,[64],num_classes,dropout,my_GraphConvolution10,n)                                          
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=0.001)
    elif model_name== '[0.9AH 0.1H]':
        dropout=0.2
        model=my_GCN(feature_size,[64],num_classes,dropout,my_GraphConvolution9,n)   
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=0.001)
    elif model_name== '[AH H]':
        dropout=0.4
        model=my_GCN(feature_size,[512, 128],num_classes,dropout,my_GraphConvolution11,n)  
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=0.005)
    elif model_name== '[AH X]':
        dropout=0.4
        model=my_GCN(feature_size,[16, 16],num_classes,dropout,my_GraphConvolution4,n)  
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=0.001)
    
    

    # Loop over each percentage
    for percent in percentages:
        val_acc_list = []
        test_acc_list = []

        # Randomize features with the given percentage
        #features = rdm_feature(initial_features, percent).to(device)
       
        features = rdm_feature(initial_features, percent).to(device)

        # Run the training process 10 times for each percentage
        for run in range(1):
            t_total = time.time()
            bad_counter = 0
            best = 999999999
            best_epoch = 0
            acc = 0

            for epoch in range(epochs):
                # Assuming your train and validate functions take the model and features as input
                loss_tra, acc_tra = train(model, features,adj,optimizer)
                loss_val, acc_val = validate(model, features,adj,optimizer)

                if (epoch + 1) % 1 == 0:
                    print('Run:{:02d}'.format(run + 1),
                          'Epoch:{:04d}'.format(epoch + 1),
                          'train',
                          'loss:{:.3f}'.format(loss_tra),
                          'acc:{:.2f}'.format(acc_tra * 100),
                          '| val',
                          'loss:{:.3f}'.format(loss_val),
                          'acc:{:.2f}'.format(acc_val * 100))

                if loss_val < best:
                    best = loss_val
                    best_epoch = epoch
                    acc = acc_val
                    bad_counter = 0
                else:
                    bad_counter += 1

                if bad_counter == patience:
                    break

            # Assuming your test function takes the model and features as input
            test_acc = test(model, features,adj,optimizer)[1]
            val_acc_list.append(acc)
            test_acc_list.append(test_acc)

        # Calculate mean and standard deviation of validation and test accuracy
        test_acc_array = np.array(test_acc_list)
        test_mean_acc = test_acc_array.mean()
        test_std_acc = test_acc_array.std()

        model_test_mean_acc[model_name].append(test_mean_acc)
        model_test_std_acc[model_name].append(test_std_acc)


# Plotting the results
plt.figure(figsize=(10, 6))
percentages=[0.1,10,20,30,40,50,60,70,80,90,100]
for model_name in  ['AH','[A H]','AH (X=I)']:
    plt.errorbar(percentages, model_test_mean_acc[model_name], yerr=model_test_std_acc[model_name], 
                 fmt='-o', capsize=5, capthick=2, label=model_name)

plt.xlabel('Percentage of Randomization of X')
plt.ylabel('Test Accuracy')
plt.title('Pubmed')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join('/mnt/data-test/figures', 'pubmed_X_01.png'), format='png', dpi=300)
