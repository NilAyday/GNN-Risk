from utils import *
import argparse

import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
import torch.optim as optim


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alignment.classes import *
from alignment.utils import *
from utils import *

from alignment.measures.randomizations import *
from alignment.measures.subspaces import *
from alignment.exceptions import ScanningCountNotPossible

import networkx as nx

from sklearn.decomposition import PCA
import scipy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
args, unknown = parser.parse_known_args()


if args.dataset=="wikipedia2":
   
    A, X, labels,idx_train,idx_val,idx_test = load_data(args.dataset)
else:
    A, X, labels,idx_train,idx_val,idx_test = load_citation(args.dataset)
    A = A.coalesce()

    
num_rdm=1
num_k=5
num_scanning=2
norm_type="Frobenius-Norm"
log=True
heatmap=True
       
    
def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
Y=encode_onehot(labels.numpy().astype(np.float32))

A=A.to_dense().numpy()

def find_repeat(source, elmt): 
    """Helper function, find index of repreat elements in source."""
    elmt_index = []
    s_index = 0;e_index = len(source)
    while(s_index < e_index):
        try:
            temp = source.index(elmt, s_index, e_index)
            elmt_index.append(temp)
            s_index = temp + 1
        except ValueError:
            break
            
    return elmt_index
def subspace_eigendecomposition(A, k):
    """Subspace of graph with dimension k"""

    vals, vecs = scipy.linalg.eigh(A, subset_by_index=(A.shape[0]-k, A.shape[0]-1))
    vals_unique_sorted = sorted(list(set(vals)), reverse=True)

    vecs_ordered = []
    for i in vals_unique_sorted:
        index_temp = find_repeat(list(vals), i)
        for j in index_temp:
            vec = vecs[:,j]
            vecs_ordered.append(vec)

    return np.array(vecs_ordered).transpose().reshape(A.shape[0], k)

def subspace_pca(X, k):
    """Subspace of a matrix by PCA where k is the dimension of subspace"""
    
    pca = PCA(n_components=k)
    pca.fit(X)
    return pca.transform(X)

def chordal(angles):
    """Chordal distance"""

    return np.linalg.norm(np.sin(angles))

def prinAngles(A, B):
    """Principal angles between two subspaces A and B"""

    Q_A, R_A = scipy.linalg.qr(A, mode='economic')
    Q_B, R_B = scipy.linalg.qr(B, mode='economic')

    U, C, Vh = scipy.linalg.svd(np.dot(Q_A.transpose(),Q_B), full_matrices=False)

    angles = np.arccos(np.clip(C, -1., 1.))
    angles.sort()

    return angles
    
def distance(X, A, Y, k_X, k_A, k_Y, norm_type):
    """Compute distance among spaces of features, graph and ground truth"""

    subspace_X = subspace_pca(X, k_X)
    subspace_A = subspace_eigendecomposition(A, k_A)
    subspace_Y = subspace_pca(Y, k_Y)

    angles_X_A = prinAngles(subspace_X, subspace_A)
    angles_X_Y = prinAngles(subspace_X, subspace_Y)
    angles_A_Y = prinAngles(subspace_A, subspace_Y)

    d_X_A = chordal(angles_X_A)
    d_X_Y = chordal(angles_X_Y)
    d_A_Y = chordal(angles_A_Y)

    # distance_matrix = np.array([[0, d_X_A, d_X_Y],
    #                             [d_X_A, 0, d_A_Y],
    #                             [d_X_Y, d_A_Y, 0]])

    if norm_type == "Frobenius-Norm":
        d_X_A_Y = np.sqrt(2 * (np.power(d_X_A,2) + np.power(d_X_Y,2) + np.power(d_A_Y,2)))
    elif norm_type == "L1-Norm":
        d_X_A_Y = 2 * (d_X_A + d_X_Y + d_A_Y)
    else:
        
        print("There is no such a norm {} as choice")
       

    return d_X_A_Y

k_Y = Y.shape[1]

opt_results = {'k_X': k_Y, 'k_A': k_Y, 'k_Y': k_Y}
G = nx.from_numpy_array(A)
    

node_list = list(G.nodes)

print("problem?")
igds = Ingredients(nodelist=node_list, G=G, X=X, A=A, Y=Y)
for idx in range(num_scanning):
    print("Scanning round {}".format(idx+1))
    if idx == 0:
        k_X_l=[int(x) for x in np.linspace(Y.shape[1], X.shape[1]-1, num=num_k)]
        k_A_l=[int(x) for x in np.linspace(Y.shape[1], A.shape[1]-1, num=num_k)]
    else:
        k_X_opt_index = k_X_l.index(opt_results['k_X'])
        if k_X_opt_index == 0:
            k_X_l_min = opt_results['k_X']
            k_X_l_max = k_X_l[1]
        elif k_X_opt_index == num_k - 1:
            k_X_l_min = k_X_l[-2]
            k_X_l_max = opt_results['k_X']
        else:
            k_X_l_min = k_X_l[k_X_opt_index-1]
            k_X_l_max = k_X_l[k_X_opt_index+1]

        k_A_opt_index = k_A_l.index(opt_results['k_A'])
        if k_A_opt_index == 0:
            k_A_l_min = opt_results['k_Y']
            k_A_l_max = k_A_l[1]
        elif k_Y_opt_index == num_k - 1:
            k_A_l_min = k_A_l[-2]
            k_A_l_max = opt_results['k_Y']
        else:
            k_A_l_min = k_A_l[k_A_opt_index-1]
            k_A_l_max = k_A_l[k_A_opt_index+1]

        k_X_l=[int(x) for x in np.linspace(k_X_l_min, k_X_l_max, num=num_k)]
        k_A_l=[int(x) for x in np.linspace(k_A_l_min, k_A_l_max, num=num_k)]

    print("k_X_l: " + str(k_X_l))
    print("k_A_l: " + str(k_A_l))

    df = pd.DataFrame(columns=['k_X','k_A','k_Y','d_zero_rdm','d_full_rdm','d_diff_zero_full_rdm'])
    i = 0

    for k_X in k_X_l:
        for k_A in k_A_l:
            d_zero_rdm = distance(
                                X=igds.get_X_gcn(p=0), 
                                A=igds.get_A_gcn(p=0), 
                                Y=igds.get_Y_gcn(), 
                                k_X=k_X, 
                                k_A=k_A, 
                                k_Y=k_Y,
                                norm_type=norm_type
                                )
            d_full_rdm = 0
            for j in range(num_rdm):
                d_full_rdm_temp = distance(
                                        X=igds.get_X_gcn(p=100), 
                                        A=igds.get_A_gcn(p=100), 
                                        Y=igds.get_Y_gcn(), 
                                        k_X=k_X, 
                                        k_A=k_A, 
                                        k_Y=k_Y,
                                        norm_type=norm_type
                                        )
                if log == True:
                        print("k_X={},k_A={},d_zero_rdm={},random_id={},d_full_rdm_temp={}".format(
                            k_X,
                            k_A,
                            d_zero_rdm,
                            j,
                            d_full_rdm_temp))
                d_full_rdm = d_full_rdm + d_full_rdm_temp
                
            d_full_rdm = d_full_rdm / num_rdm
            df.loc[i] = [k_X,k_A,k_Y,d_zero_rdm,d_full_rdm,d_full_rdm-d_zero_rdm]
            i = i + 1

        
opt_results = dict(df.iloc[df['d_diff_zero_full_rdm'].idxmax()].astype(int)[["k_X","k_A","k_Y"]])
print("Optimization results in scanning round {}: {}".format(idx+1,opt_results))
print("\n")

   


    
