import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
from process import *
import pickle
import itertools
import pickle
import os
import argparse
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--model', type=str, default='gcn_1', help='Which model to train')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=10, help='patience')
parser.add_argument('--runs', type=int, default=10, help='number of times the exp run.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--nhid_list', type=str, default='[]', help='Hidden dim list.')

#args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

#args.nhid_list= [float(i) for i in args.nhid_list.split(',')] 
args.nhid_list = [int(float(i)) for i in args.nhid_list.replace(' ', '').split(',')]
#args.nhid_list = str([int(float(i)) for i in args.nhid_list.replace(' ', '').split(',') if i]) if args.nhid_list else args.nhid_list

#if ',' in args.nhid_listeight_decay['nhid_list'] else []

print(args.nhid_list)
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.dataset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features = features.to(device)
adj = adj.to(device).coalesce()
n=features.shape[0]
nfeat=features.shape[1]
nclass = len(torch.unique(labels))

checkpt_file = '/mnt/data-test/pretrained/'+uuid.uuid4().hex+'.pt'



if args.model == "model_1":
    conv_layer = my_GraphConvolution1
elif args.model == "model_2":
    conv_layer = my_GraphConvolution2
elif args.model == "model_3":
    conv_layer = my_GraphConvolution3
elif args.model == "model_4":
    conv_layer = my_GraphConvolution4
elif args.model == "model_5":
    conv_layer = my_GraphConvolution5
elif args.model == "model_6":
    conv_layer = my_GraphConvolution6
elif args.model == "model_7":
    conv_layer = my_GraphConvolution7
elif args.model == "model_8":
    conv_layer = my_GraphConvolution8
elif args.model == "model_9":
    conv_layer = my_GraphConvolution9  
elif args.model == "model_10":
    conv_layer = my_GraphConvolution10
elif args.model == "model_11":
    conv_layer = my_GraphConvolution11
else:
    raise ValueError("Invalid model type specified")



results = []

val_acc_list = []
test_acc_list = []

    
for run in range(args.runs):
        t_total = time.time()
        bad_counter = 0
        best = float('inf')
        best_epoch = 0
        best_val_acc = 0
        
        model = my_GCN(nfeat, args.nhid_list, nclass, args.dropout, conv_layer,n).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(args.epochs):
           
            loss_tra, acc_tra = train(model, optimizer, features, adj, labels,idx_train,device)
            loss_val, acc_val = validate(model, features, adj, labels,idx_val,device)
            

            if (epoch + 1) % 10 == 0:
                print('Run:{:02d}'.format(run+1),
                      'Epoch:{:04d}'.format(epoch+1),
                      'train',
                      'loss:{:.3f}'.format(loss_tra),
                      'acc:{:.2f}'.format(acc_tra * 100),
                      '| val',
                      'loss:{:.3f}'.format(loss_val),
                      'acc:{:.2f}'.format(acc_val * 100))

            if loss_val < best:
                best = loss_val
                best_epoch = epoch
                best_val_acc = acc_val
                torch.save(model.state_dict(), checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break
        
        val_acc_list.append(best_val_acc)
        test_acc = test(model, features, adj, labels,idx_test,checkpt_file,device)[1]
        test_acc_list.append(test_acc)

results.append({
    'run': run + 1,
    'val_acc_mean': np.mean(val_acc_list),
    'val_acc_std': np.std(val_acc_list),
    'test_acc_mean': np.mean(test_acc_list),
    'test_acc_std': np.std(test_acc_list),
})

    # Create the directory if it does not exist
results_dir = '/mnt/data-test/results'
os.makedirs(results_dir, exist_ok=True)

    # Construct the file name
file_name = f'{args.model}_dataset={args.dataset}_nhid={args.nhid_list}_dropout={args.dropout}_epochs={args.epochs}_lr={args.lr}_wd={args.wd}_patience={args.patience}_runs={args.runs}.pkl'

file_path = os.path.join(results_dir, file_name)

with open(file_path, 'wb') as f:
    pickle.dump(results, f)

