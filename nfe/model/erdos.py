import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, GatedGraphConv, GCNConv, SAGEConv, EGConv, GATConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import degree
import torch_geometric
import numpy as np
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean

from model.set_functions_loader import get_set_functions

import networkx as nx
from torch_geometric import utils
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops

from extensions.extension_loader import get_extension_functions
import random
#from nfe.baselines import greedy_maximization
from utils.laplacian import handle_lap
from utils.signnet import get_sign_inv_net, GINDeepSigns

import extensions.neural as neural
import extensions.lovasz as lovasz
from torch.distributions import relaxed_categorical, Categorical
from data.gurobi import self_implement_degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ErdosNet(torch.nn.Module):
    def __init__(self, num_layers, hidden1, args):
        super(ErdosNet, self).__init__()
        self.args=args
        self.hidden1 = hidden1
        self.convs = torch.nn.ModuleList()
        self.numlayers = num_layers
        self.heads = 1
        self.conv  = GatedGraphConv(self.hidden1, self.numlayers)
        self.lin1 = Linear(self.hidden1, self.hidden1)
        self.lin2 = Linear(self.hidden1, 1)
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
        
####TODO: make losses work through function dict
    def maxcut_loss(self,probs, edge_index, batch, degrees):
        row, col = edge_index
        expected_weight =  scatter_add((probs[row]*probs[col]), batch[row], 0)
        expected_degree =  scatter_add(probs*degrees, batch, 0)
        expected_cut =  expected_degree - expected_weight
        return -expected_cut/2.   

    def maxclique_loss(self,probs,edge_index,  batch, degrees):
        row, col = edge_index
        num_graphs = batch.max()+1 
            #calculating the terms for the expected distance between clique and graph
        pairwise_prodsums = torch.zeros(num_graphs, device = device)
        for graph in range(num_graphs):
            batch_graph = (batch==graph)
            pairwise_prodsums = (torch.conv1d(probs[batch_graph].unsqueeze(-1), probs[batch_graph].unsqueeze(-1))).sum()/2
        ###calculate loss terms
        self_sums = scatter_add((probs*probs), batch, 0, dim_size = num_graphs)
        expected_weight_G = scatter_add(probs[row]*probs[col], batch[row], 0, dim_size = num_graphs)/2.
        expected_clique_weight = (pairwise_prodsums.unsqueeze(-1) - self_sums)/1.
        expected_distance = (expected_clique_weight - expected_weight_G)        
        penalty_coefficient = self.args.erdos_penalty
        ###calculate loss
        expected_loss = (penalty_coefficient)*expected_distance*0.5 - 0.5*expected_weight_G  
        return expected_loss 
    
    def cliqueness(self,indicator,edge_index):
        row, col = edge_index
        weight =  (indicator[row]*indicator[col]).sum()/2.
        num_nodes = indicator.sum()
        if num_nodes==1:
            return True
        max_weight = num_nodes*(num_nodes-1)*0.5
        #print("weight: ", weight)
        #print("max weight: ", max_weight)
        if (weight/max_weight)>=1.:
            return True
        else:    
            return False
    
    def weight(self, indicator, edge_index):
        #has to be undirected
        row, col = edge_index
        weight =  (indicator[row]*indicator[col]).sum()/2.
        return weight

    def cardinality(self, indicator, edge_index):
        card = indicator.sum()
        return card
    
    def cut(self,indicator, edge_index):
        row,col = edge_index
        cut = (torch.abs(indicator.T[row]-indicator.T[col])).sum()/2.
        return cut

        

    def independence(self, indicator, edge_index):
        row, col = edge_index
        weight =  (indicator[row]*indicator[col]).sum()/2.
        if weight>0.:
            return False
        else:
            return True

    def maxindset_loss(self, probs, edge_index, batch, degrees):
        row, col = edge_index
        expected_cardinality = scatter_add(probs, batch, 0) 
        expected_weight =  scatter_add((probs[row]*probs[col]), batch[row], 0)
        penalty_coefficient = self.args.erdos_penalty
        loss = -expected_cardinality +penalty_coefficient*expected_weight
        return loss
        # TODO



    def compute_loss(self, probs, edge_index, batch, problem_name, degrees):
        if 'clique' in problem_name:
            loss = self.maxclique_loss(probs,edge_index, batch,degrees)
        elif "max_indep_set" in problem_name:
            loss = self.maxindset_loss(probs, edge_index, batch, degrees)
        elif "cut" in problem_name:
            loss =  self.maxcut_loss(probs, edge_index, batch, degrees)
        return loss

    def erdos(self, x, function_dict, penalty): 
        n_sets = self.get_n_sets(x)
        f_sets=torch.zeros(n_sets).to(device)
        f_unreg_sets=torch.zeros(n_sets).to(device)
        

        min_set_val = f_unreg_sets.min()

        return f_sets, min_set_val

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

    def unconstrained(self, indicator, edge_index):
        return True



    def greedy_decode(self, x, edge_index, function_dict, penalty):
        indices = x.argsort()
        num_nodes = x.shape[0]
        if "clique" in self.args.problem:
            check_constraint = self.cliqueness
            objective = self.weight
        elif "indep_set" in self.args.problem:
            check_constraint = self.independence
            objective = self.cardinality
        elif "cut" in self.args.problem:
            check_constraint = self.unconstrained
            objective = self.cut

        solution = torch.zeros_like(x)
        curr_objective = objective(solution, edge_index)
        for k in indices:
            solution[k] = 1.
            #breakpoint()
            if objective(solution, edge_index)<curr_objective:
                solution[k] = 0.
                #break
            else:
                curr_objective = objective(solution, edge_index)
            if not check_constraint(solution, edge_index):
                solution[k] = 0.
                #break
            
        return solution

    #slower
    def cond_expect(self, x, edge_index, function_dict, penalty, batch):
        indices = x.argsort()
        num_nodes = x.shape[0]
        r,c = edge_index
        degrees = degree(r)
        if "clique" in self.args.problem:
            check_constraint = self.cliqueness
            objective = self.maxclique_loss
        elif "indep_set" in self.args.problem:
            check_constraint = self.independence
            objective = self.maxindset_loss
        elif "cut" in self.args.problem:
            check_constraint = self.unconstrained
            objective = self.maxcut_loss

        breakpoint()
        solution = x.detach()   
        curr_objective = objective(solution, edge_index, batch, degrees)
        solution_indices =[]
        for counter, k in enumerate(indices):
            solution[k] = 1.
            #breakpoint()
            if objective(solution, edge_index)<curr_objective:
                solution[k] = 0.
            else:
                curr_objective = objective(solution, edge_index, batch, degrees)
                solution_indices += [k]
            if not check_constraint(solution, edge_index):
                solution[k] = 0.
                break
        
        solution = torch.zeros_like(x)
        solution[solution_indices] = 1.

            
        return solution
    



    def sample_and_evaluate(self, x, function_dict, penalty):
        num_samples = self.args.num_erdos_samples
        sets = ((x>torch.rand((x.shape[0], num_samples),device='cuda'))*1.).T
        #breakpoint()
        f_sets, f_unreg_sets = self.set_function(sets, function_dict, self.args, penalty)
        #breakpoint()
        min_set_val = f_unreg_sets.min()

        return f_sets, f_unreg_sets, min_set_val

    def fix_batch_degrees(self, data):
        full_degs = []
        for graph in data.to_data_list():
            r,c = graph.edge_index
            degrees =  degree(r,graph.num_nodes)
            full_degs += [degrees]

        #breakpoint()
        full_degs = torch.cat(full_degs)
        return full_degs


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

    
        if not self.args.extension in ['random_walk', 'karger', 'neural']:
            #this block maps the embedding of each node to a scalar. If using x for random walk/Kargers
            #then we still need a vector value for each node, so we do not apply this block.
            x = F.leaky_relu(x) 
            x = (self.lin2(x))
            x = torch.sigmoid(x)



    
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

        #calculate the loss on the full batch first
        no_loop_index,_ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index

        degrees = self.fix_batch_degrees(data)

        loss = self.compute_loss(x, no_loop_index , data.batch, args.problem, degrees)
    
        #breakpoint()
        full_degs = []
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

            solution_set = self.greedy_decode(graph_x.detach(), graph.edge_index, function_dict, args.penalty)
            

            f_sets, f_unreg_sets = self.set_function(solution_set.detach(), function_dict, args.penalty)
            #breakpoint()
            min_set_vals[counter] = f_unreg_sets.min()
 
        loss = loss.mean()



        output = {  "x": x,
                    "best_sets": min_set_vals,
                    "loss": loss}
        
        return output

     



