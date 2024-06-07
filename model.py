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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

# [AH,A,X]
class my_GraphConvolution1(nn.Module):
    def __init__(self, in_features, out_features, nfeat, n, bias=True):
        super(my_GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features + n + nfeat, out_features))
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

    def forward(self, input, adj, x):
        support = torch.cat((torch.mm(adj.to_dense(), input), adj.to_dense(), x), 1)
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# [AH+H]
class my_GraphConvolution2(nn.Module):
    def __init__(self, in_features, out_features, nfeat, n, bias=True):
        super(my_GraphConvolution2, self).__init__()
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

    def forward(self, input, adj, x):
        support = torch.mm(adj.to_dense(), input) + input
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# [X,AH+H]
class my_GraphConvolution3(nn.Module):
    def __init__(self, in_features, out_features, nfeat, n, bias=True):
        super(my_GraphConvolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(nfeat + in_features, out_features))
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

    def forward(self, input, adj, x):
        support = torch.cat((x, torch.mm(adj.to_dense(), input) + input), 1)
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# [AH,X]
class my_GraphConvolution4(nn.Module):
    def __init__(self, in_features, out_features, nfeat, n, bias=True):
        super(my_GraphConvolution4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(nfeat + in_features, out_features))
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

    def forward(self, input, adj, x):
        support = torch.cat((torch.mm(adj.to_dense(), input), x), 1)
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class my_GCN(nn.Module):
    def __init__(self, nfeat, nhid_list, nclass, dropout, conv_layer, n):
        super(my_GCN, self).__init__()
        self.layers = nn.ModuleList()
        
        if nhid_list:
            # Input layer
            self.layers.append(conv_layer(nfeat, nhid_list[0], nfeat, n))
            # Hidden layers
            for i in range(1, len(nhid_list)):
                self.layers.append(conv_layer(nhid_list[i-1], nhid_list[i], nfeat, n))
            # Output layer
            self.layers.append(conv_layer(nhid_list[-1], nclass, nfeat, n))
        else:
            # Single output layer
            self.layers.append(conv_layer(nfeat, nclass, nfeat, n))
        
        self.dropout = dropout

    def forward(self, x, adj):
        h = x.detach().requires_grad_()
        
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h, adj, x))
            h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.layers[-1](h, adj, x)
        
        return F.log_softmax(h, dim=1)
