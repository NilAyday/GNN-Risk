import torch
from torch_geometric.nn import GCNConv, GATConv, ChebConv,  JumpingKnowledge

from torch_geometric.utils import add_self_loops, get_laplacian
from torch.nn import Sequential, Linear, ReLU,ModuleList
from torch_geometric.nn.dense.linear import Linear as dense_Linear
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree, remove_self_loops,to_undirected
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing

import numpy as np
import math
from typing import Optional

import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


from typing import List, Optional
from torch import Tensor

from torch_sparse import SparseTensor

#https://github.com/tkipf/pygcn/tree/master
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
        
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class my_GraphConvolution(nn.Module):
    

    def __init__(self, in_features, out_features, bias=True):
        super(my_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        
        
        #support = torch.cat((input-torch.mm(adj.to_dense(), input), adj.to_dense()),1)
        
        #support = torch.cat((input, adj.to_dense()),1)
        #output = torch.mm(support, self.weight)

        output= torch.mm(torch.mm(adj.to_dense(), input) + input, self.weight)

       
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
        
class my_GCN(nn.Module):
    def __init__(self,n, nfeat, nhid, nclass, dropout):
        super(my_GCN, self).__init__()

        self.gc1 = my_GraphConvolution(int(round(nfeat)) ,nhid)
        #self.gc1 = my_GraphConvolution(int(round(nfeat + n)) , int(round((nfeat + n)/2)))
        self.gc2 = my_GraphConvolution(nhid , nclass)
        
        self.dropout = dropout

    def forward(self, x, adj):
        #x = self.gc1(x, adj)
        
        h=x.detach().requires_grad_()
        h = F.relu(self.gc1(h, adj))
        h = F.dropout(h, self.dropout, training=self.training)
    
        h = self.gc2(h, adj)
     
        return F.log_softmax(h, dim=1)

