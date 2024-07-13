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

import matplotlib.pyplot as plt
import os
import pickle



def rdm_graph(G, nodelist, percent):
    """Reshuffle graph edges with percentage p (p>0)."""
    
    n = len((G.edges()))

    # Sort each edge
    edgelist = list(G.edges())
    edgelist = [(int(i),int(j)) for (i,j) in edgelist]

    # Sample a subset of edges from edgelist with the rate p
    sample = random.sample(list(G.edges()), int(n*percent/100))
    sample = [(int(i),int(j)) for (i,j) in sample]
    
    sample_sorted = sorted(sample, key=lambda tup: (tup[0],tup[1]))

    unchanged_edgelist = list(set(edgelist).difference(set(sample)))

    node1_list, node2_list = zip(*sample_sorted)
    node_list = list(node1_list + node2_list)
    counter=collections.Counter(node_list)
    node_dict = dict(counter)

    G_sample_random = nx.configuration_model(list(node_dict.values()))

    node_label_mapping = {v: k for v, k in enumerate(list(node_dict.keys()))}
    random_edges = list(G_sample_random.edges())
    random_edges_after_mapping = [(node_label_mapping[i],node_label_mapping[j]) for (i,j) in random_edges]

    random_edges_after_mapping_new = []
    for (i,j) in random_edges_after_mapping:
        if i > j:
            m = i
            i = j
            j = m
        random_edges_after_mapping_new.append((i,j))

    random_edges_after_mapping_new_sorted = sorted(random_edges_after_mapping_new, key=lambda tup: (tup[0],tup[1]))
    new_edgelist = list(unchanged_edgelist + random_edges_after_mapping_new_sorted)
    new_edgelist_sorted = sorted(new_edgelist, key=lambda tup: (tup[0],tup[1]))

    G_new = nx.MultiGraph()
    G_new.add_edges_from(new_edgelist_sorted)

    G_new = nx.Graph(G_new)
    nb_multi_edges = len(new_edgelist_sorted) - len(G_new.edges())

    G_new.remove_edges_from(list(nx.selfloop_edges(G_new)))
    nb_self_loops = len(new_edgelist_sorted) - len(G_new.edges()) - nb_multi_edges
    
    return torch.tensor(np.array(nx.adjacency_matrix(G_new, nodelist=nodelist).toarray(),dtype=np.float32)).to(device)

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


models =  ['AH','[0.8*AH 0.1*H 0.1*X]','AH (A=I)','[0.9AH 0.1H]']
#['AH','[0.8*AH 0.2*A]','AH (A=I)']# Replace with your actual model names or instances
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
    elif model_name== 'AH (A=I)':
        dropout=0.5
        model=my_GCN_A_I(feature_size,[256],num_classes,dropout,my_GraphConvolution13,n)
        model=model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=0.005)
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
    
    

    # Loop over each percentage
    for percent in percentages:
        val_acc_list = []
        test_acc_list = []

        # Randomize features with the given percentage
        #features = rdm_feature(initial_features, percent).to(device)
       
        adj=rdm_graph(G,  list(G.nodes),percent)

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
for model_name in ['AH','[0.8*AH 0.1*H 0.1*X]','AH (A=I)','[0.9AH 0.1H]']:
    plt.errorbar(percentages, model_test_mean_acc[model_name], yerr=model_test_std_acc[model_name], 
                 fmt='-o', capsize=5, capthick=2, label=model_name)

plt.xlabel('Percentage of Randomization of A')
plt.ylabel('Test Accuracy')
plt.title('Pubmed')
plt.legend()
plt.grid(True)

'''
# Save the figure as a pickle file
with open('/mnt/data-test/figures/test_accuracy_plot.pkl', 'wb') as f:
    pickle.dump(plt.gcf(), f)
'''
plt.savefig(os.path.join('/mnt/data-test/figures', 'pubmed_A_1.png'), format='png', dpi=300)
