import torch
import torch_geometric

import networkx as nx
from extensions.utils.wilson import UST

import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_for_sampling(x, function_dict, args): 

    x = x @ x.T / np.sqrt(x.shape[0])
    x = F.softmax(x, dim=1)

    max_val=args.max_val
    #x = F.normalize(x, p=2, dim=1)
    #edge_scores = x @ x.T
    #x_sum       = torch.norm(x, p=2, keepdim=True)
    #edge_scores = edge_scores / (x_sum @ x_sum.T)
    #edge_scores = torch.sigmoid(edge_scores)

    graph_nx=torch_geometric.utils.to_networkx(function_dict["graph"], to_undirected=function_dict["is_undirected"])

    ust = UST(graph_nx, x)

    for _ in range(args.n_sets):
        ust.sample('Wilson')

    spanning_trees = ust.list_of_samples

    return spanning_trees, x
        
def sample_set(i, sampling_data, args):
    spanning_trees, x = sampling_data
    tree=spanning_trees[i]
    new_set = cut_set_from_tree(tree, x)
    prob=prob_of_tree(tree, x)

    return new_set, prob

### KARGER (VIA WILSONS) UTILITY FUNCTIONS ###       
def cut_set_from_tree(tree, edge_scores):
    tree_orig=tree
    tree=list(tree.edges())

    tree_edge_probs={edge: edge_scores[edge] for edge in tree}

    max_edge=max(tree_edge_probs, key=lambda key: tree_edge_probs[key])

    tree_orig.remove_edge(*max_edge)
    cut_indices = torch.tensor(list(list(nx.connected_components(tree_orig))[0]))
    
    cut = torch.zeros(len(tree_orig.nodes()))
    cut[cut_indices] = 1
    return cut

def prob_of_tree(tree, edge_scores):
    tree=list(tree.edges())

    if len(tree)>0:
        tree=tree[:6]
        prob=sum([edge_scores[edge].log() for edge in tree]).exp()
        #prob=sum([edge_scores[edge] for edge in tree])/len(tree)
    else:
        prob=torch.tensor([0.], requires_grad=True).to(device)

    return prob