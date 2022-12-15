import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, GatedGraphConv, GCNConv, SAGEConv, EGConv, GATConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import degree
import torch_geometric
import numpy as np
from torch_geometric.nn import GlobalAttention
from torch.nn import TransformerEncoderLayer

from model.set_functions_loader import get_set_functions

import networkx as nx
from torch_geometric import utils

from extensions.extension_loader import get_extension_functions
import random
from utils.laplacian import handle_lap
from utils.signnet import get_sign_inv_net, GINDeepSigns

import extensions.neural as neural
import extensions.lovasz as lovasz
from torch.distributions import relaxed_categorical, Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReinforceBaselineNet(torch.nn.Module):
    def __init__(self, num_layers, hidden1, args):
        super(ReinforceBaselineNet, self).__init__()
        self.args=args
        self.hidden1 = hidden1
        self.convs = torch.nn.ModuleList()
        self.numlayers = num_layers
        self.heads = 1
        self.conv  = GatedGraphConv(self.hidden1, self.numlayers)

        self.linfirst = Linear(args.input_feat_dim, self.hidden1)

        self.lin1 = Linear(self.hidden1, self.hidden1)
        self.lin2 = Linear(self.hidden1, 2)
        self.num_samples = args.num_reinforce_samples
        self.preprocess_for_sampling, self.sample_set = get_extension_functions(self.args.extension)
        self.set_function = get_set_functions(self.args.extension, args)

        self.deterministic=True
        self.lap_method = args.lap_method
        if args.lap_method is not None:
            self.SignNet=get_sign_inv_net(hidden1, args)

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
        self.linfirst.reset_parameters()
        #self.global_attention.reset_parameters()
        #self.global_attention_tformer.reset_parameters()

    def reinforce_with_baseline(self, x, function_dict, baseline, penalty): 
        
        #sampling_data=function_dict["sampling_data"]    
        #sampling_data = self.preprocess_for_sampling(x, function_dict, self.args)

        #breakpoint()
        n_sets = self.get_n_sets(x)
        # sets=n_sets*[0.]
        # probs=n_sets*[0.]
        f_sets=torch.zeros(n_sets).to(device)
        f_unreg_sets=torch.zeros(n_sets).to(device)
        
        #SAMPLE SETS here
        log_prob_list = []

        num_samples = self.num_samples
        cat = Categorical(logits = x)
        sets  = cat.sample(sample_shape=(num_samples,))
        log_prob_action = cat.log_prob(sets)
        log_prob_action = log_prob_action.T.sum(0)
        #breakpoint()
        rewards, f_unreg_sets = self.set_function(sets, function_dict, self.args.penalty)

        #breakpoint()
        #reward normalization
        #breakpoint()


        ###BASELINE
        baseline_distribution = Categorical(probs=torch.ones_like(x)*0.5)
        baseline_sets = baseline_distribution.sample(sample_shape=(num_samples,))
        log_prob_action_baseline = baseline_distribution.log_prob(sets)
        log_prob_action_baseline = log_prob_action_baseline.T.sum(0)
        
        

        ###
        #breakpoint()
        baseline_rewards, _ = self.set_function(baseline_sets, function_dict, self.args.penalty)


        #breakpoint()


        if "indep" in self.args.problem:
            #rewards = rewards-graph
            rewards = rewards - baseline_rewards.mean()
            normed_reward = rewards
            #normed_reward  = rewards - rewards.mean()
            # normed_reward = normed_reward/(normed_reward.std()+(1e-6))
            #normed_reward = (rewards<rewards.mean())*1.
            #print("here")
        else:
            rewards = rewards-baseline_rewards.mean()
            # normed_reward  = rewards - rewards.mean()
            # normed_reward = normed_reward/(normed_reward.std()+(1e-6))
            
        # if(f_unreg_sets.sum()<=2.):
        #     print("rewards:", rewards)
        #     print("normed rewards: ", normed_reward)

        min_set_val = f_unreg_sets.min()
        #print(min_set_val)
        #breakpoint()
        return normed_reward, log_prob_action, min_set_val

    def get_n_sets(self, x):
        if self.args.extension in ['lovasz', 'lovasz_old', 'neural', 'lovasz_fixed_cardinality']:
            if self.args.extension=='neural' and self.args.neural=='v3':
                n_sets=self.args.n_sets
            elif self.args.extension=='lovasz_fixed_cardinality':
                n_sets = max(len(x) - (self.args.k_clique_no-1), 1)
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
        #breakpoint()
        #x = self.global_attention_tformer(x)
        #x = self.global_attention(x.unsqueeze(-1))
        #x = F.leaky_relu(self.linfirst(x))
        x =  F.leaky_relu(self.conv1(x, edge_index)) #+x
        x =  F.leaky_relu(self.conv(x, edge_index))  +x

        #linear
        x = self.lin1(x)

    
        if not self.args.extension in ['random_walk', 'karger', 'neural']:
            #this block maps the embedding of each node to a scalar. If using x for random walk/Kargers
            #then we still need a vector value for each node, so we do not apply this block.
            x = F.leaky_relu(x) 
            x = self.lin2(x)
            if "indep_set" in self.args.problem:
                x = torch.sigmoid(x)*0.95 +torch.rand_like(x)*1e-4



    
        #prepare for extension
        num_graphs = data.batch.max().item() + 1    
        extension_values = num_graphs*[0.]
        min_set_vals = num_graphs*[0.]

        ####NEW
        #eigenvalues, all_eigenvectors = self.preprocess_for_sampling(x, None, self.args)
        #all_eigenvectors = self.NeuralSignNet(all_eigenvectors, edge_index)
        #all_eigenvectors = F.normalize(all_eigenvectors, dim=0)

        #print("I'm IN HERE!")
        ##breakpoint()
        log_prob_list = []
        loss = 0
        for counter, graph in enumerate(data.to_data_list()):

            graph_xs = x[data.batch==counter]     
            graph_x=graph_xs.squeeze()

            #breakpoint()
            mr, mc =  graph.edge_index
            numels = graph_x.shape[0]



            function_dict = {"function_name": args.problem, 
                                "graph": graph, 
                                #"sampling_data": sampling_data,
                                "is_undirected": True, 
                                "cardinality_const": args.cardinality_const}
            #breakpoint()

            rewards, log_probs, min_set_vals[counter] =  self.reinforce_with_baseline(graph_x,function_dict, graph.baseline ,args.penalty)
            log_prob_list += [log_probs]
            

            loss += (log_probs*(rewards)).sum()
        #breakpoint()
        #breakpoint()
        output = {  "x": x,
                    "best_sets": min_set_vals,
                    "loss": loss}
        
        return output

     



