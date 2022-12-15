import os 
import pickle

import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, GatedGraphConv, GCNConv, SAGEConv, EGConv, GATConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import degree
import torch_geometric
import numpy as np

from model.set_functions_loader import get_set_functions

import networkx as nx
from torch_geometric import utils

from extensions.extension_loader import get_extension_functions
import random
from utils.laplacian import handle_lap
from utils.signnet import get_sign_inv_net, GINDeepSigns

import extensions.neural as neural
import extensions.lovasz as lovasz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NonnegativeNet(torch.nn.Module):
    def __init__(self, num_layers, hidden1, args):
        super(NonnegativeNet, self).__init__()
        self.args=args
        self.hidden1 = hidden1
        self.convs = torch.nn.ModuleList()
        self.numlayers = num_layers
        self.heads = 1
        self.conv  = GatedGraphConv(self.hidden1, self.numlayers)
        self.lin1 = Linear(self.hidden1, self.hidden1 if self.args.extension!='lovasz_multi' else self.args.n_sets)
        self.lin2 = Linear(self.hidden1, args.output_sets if args.problem!='shortest_path' else self.hidden1)

        self.preprocess_for_sampling, self.sample_set = get_extension_functions(self.args.extension)
        self.set_function = get_set_functions(self.args.extension, args)

        self.deterministic=True
        self.lap_method = args.lap_method
        if args.lap_method is not None:
            self.SignNet=get_sign_inv_net(hidden1, args)

        self.eig_data=[]
        #if self.args.extension=='neural':
        #    self.NeuralSignNet=GINDeepSigns(1, hidden1, hidden1, gin_layers=2, mlp_layers=3, mlp_out_dim=self.args.n_sets, use_bn=True, dropout=False, activation='relu')
        #else:
        #    self.NeuralSignNet=None


        if args.base_gnn=='gat':
            self.conv1 = GATConv(args.input_feat_dim, self.hidden1)
        if args.base_gnn=='gcn':
            self.conv1 = GCNConv(args.input_feat_dim, self.hidden1)
        if args.base_gnn=='egc': 
            self.conv1 = EGConv(args.input_feat_dim, self.hidden1, aggregators=['min'])            
        if args.base_gnn=='gin':
            self.conv1 = GINConv(Sequential(Linear(args.input_feat_dim, self.hidden1), ReLU(),
                Linear(self.hidden1, self.hidden1),
                ReLU(),
                BN(self.hidden1),
            ),train_eps=False)
        if args.base_gnn=='sage':
            self.conv1 = SAGEConv(args.input_feat_dim, self.hidden1)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.conv1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        

    def calculate_extension(self, x, function_dict, penalty): 

        #sampling_data=function_dict["sampling_data"]    
        sampling_data = self.preprocess_for_sampling(x, function_dict, self.args)

        if function_dict["save_evals"] is True:
            gram = x @ x.T 
            eigenvalues=torch.linalg.eigvals(gram)
            self.eig_data.append(eigenvalues)
            with open(os.path.join(self.args.save_root, f"eigvals_PROT.pkl"), 'wb') as handle:
                pickle.dump(self.eig_data, handle)

        magraph = function_dict['graph']
        #n_sets = self.get_n_sets(x)
        n_sets = x.shape[0]
        #print(n_sets)
        sets=n_sets*[0.]
        probs=n_sets*[0.]
        f_sets=torch.zeros(n_sets).to(device)
        f_unreg_sets=torch.zeros(n_sets).to(device)

        if self.args.extension in ['lovasz_old', 'neural', 'nonnegative']:
            for i in range(n_sets):
                
                #breakpoint()
                print("Nsets: ", n_sets)
                print("i: ", i)
                print("x shape", x.shape)
                #print(" ")

                sets[i], probs[i] = self.sample_set(i, sampling_data, self.args)
                
                #breakpoint()
                
                f_sets[i], f_unreg_sets[i] = self.set_function(sets[i], function_dict, self.args, penalty)

            probs = torch.cat([p.unsqueeze(0) for p in probs])
            
        else:
            sets, probs = self.sample_set(sampling_data, self.args)
            f_sets, f_unreg_sets = self.set_function(sets, function_dict, self.args, penalty)


        extension   = (f_sets*probs).sum()


        min_set_val = f_unreg_sets.min()

        #print(min_set_val)


        return extension, min_set_val

    def get_n_sets(self, x):
        if self.args.extension in ['lovasz', 'nonnegative', 'lovasz_old', 'neural', 'lovasz_fixed_cardinality', 'lovasz_multi', 'lovasz_bounded_cardinality']:
            if self.args.extension=='neural' and self.args.neural in ['v3', 'v4']:
                n_sets=self.args.n_sets
            elif self.args.extension=='lovasz_fixed_cardinality':
                n_sets = max(len(x) - (self.args.k_clique_no-1), 1)
            elif self.args.extension=='lovasz_multi':
                n_sets = x.shape[1]
            else:
                n_sets = len(x)  
        elif self.args.eig_sym is True:
            n_sets=2*self.args.n_sets

        else:
            n_sets=self.args.n_sets

        return n_sets

    def forward(self, data, args, warmup=False):
        if args.features=='lap_pe':
            x = handle_lap(data, self.SignNet, self.lap_method)
        else:
            x = data.x

        if x.dim()==1:
            x = x.unsqueeze(-1)

        edge_index= data.edge_index.detach()
        #graph convs
        #breakpoint()
        x =  F.leaky_relu(self.conv1(x, edge_index)) #+x
        x =  F.leaky_relu(self.conv(x, edge_index))  +x

        #linear
        x = self.lin1(x)
        x = torch.sigmoid(x)
    
        if not self.args.extension in ['random_walk', 'karger', 'neural', 'lovasz_multi', 'nonnegative']:
            #this block maps the embedding of each node to a scalar. If using x for random walk/Kargers
            #then we still need a vector value for each node, so we do not apply this block.
            x = F.leaky_relu(x) 
            x = F.leaky_relu(self.lin2(x)) 
            x = torch.sigmoid(x)




    
        #prepare for extension
        num_graphs = data.batch.max().item() + 1    
        extension_values = num_graphs*[0.]
        min_set_vals = num_graphs*[0.]

        ####NEW
        #eigenvalues, all_eigenvectors = self.preprocess_for_sampling(x, None, self.args)
        #all_eigenvectors = self.NeuralSignNet(all_eigenvectors, edge_index)
        #all_eigenvectors = F.normalize(all_eigenvectors, dim=0)


        for counter, graph in enumerate(data.to_data_list()):

            graph_xs = x[data.batch==counter]     
            graph_x=graph_xs.squeeze()
            #graph_x=F.softmax(graph_x, dim=0)
            #breakpoint()

            if self.args.extension=='lovasz_multi':
                graph_x = torch.sigmoid(graph_x)
                graph_x = graph_x.T


            function_dict = {"function_name": args.problem, 
                                "graph": graph, 
                                #"sampling_data": sampling_data,
                                "is_undirected": True, 
                                "cardinality_const": args.cardinality_const,
                                "save_evals": args.save_evals_epoch}
            extension_values[counter], min_set_vals[counter] = self.calculate_extension(graph_x, function_dict, penalty=args.penalty)

        loss = sum(extension_values)/len(extension_values)
        

        output = {  "x": x,
                    "best_sets": min_set_vals,
                    "loss": loss}
        
        return output

     



