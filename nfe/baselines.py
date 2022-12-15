import torch
import torch_geometric
from torch_geometric.utils import degree
from model.set_functions import call_set_function
import numpy as np
import networkx as nx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def greedy_max(data,args):
    batch = data.batch
    num_graphs = batch.max().item() + 1
    edge_index= data.edge_index.detach()
    row, col = edge_index
    fname = args.problem

    if fname=='clique_v4':
        fname=fname+'_old'

    batch_data_list = data.to_data_list()
    best_set_values = torch.zeros(num_graphs)

    for counter, graph in enumerate(batch_data_list):
        nx_graph = torch_geometric.utils.convert.to_networkx(graph).to_undirected()
        if 'clique' in args.problem:
            greedy_sol = len(nx.algorithms.approximation.maximum_independent_set(nx.complement(nx_graph)))
        if 'indep_set' in args.problem:
            greedy_sol = len(nx.algorithms.approximation.maximum_independent_set(nx_graph))
        if 'cut' in args.problem:
            greedy_sol = nx.algorithms.approximation.one_exchange(nx_graph)[0]
        best_set_values[counter] = -greedy_sol
    return best_set_values


def greedy_maximization(data, args):
    batch = data.batch
    num_graphs = batch.max().item() + 1
    edge_index= data.edge_index.detach()
    row, col = edge_index
    fname = args.problem

    if fname=='clique_v4':
        fname=fname+'_old'

    batch_data_list = data.to_data_list()
    best_set_values = torch.zeros(num_graphs)

    for counter, graph in enumerate(batch_data_list):
        terminate=False
        current_set=torch.zeros_like(graph.x)
        best_value=np.inf

        function_dict = {"function_name": fname, "graph": graph, "is_undirected": True, "cardinality_const": args.cardinality_const}

        
        while not terminate:
            unused_indices=(current_set == 0).nonzero().squeeze()
            add_index=None

            for index in unused_indices:
                proposal=current_set.clone()
                proposal[index]=1
            
                _, value = call_set_function(proposal, function_dict, args.penalty, args.k_clique_no)
                if value < best_value:
                    add_index  = index
                    best_value = value

            if add_index is None:
                terminate=True
            else:
                current_set[add_index]=1

        _, best_set_values[counter]=call_set_function(current_set, function_dict, args.penalty, args.k_clique_no)

    return best_set_values


def random_maximization(data, args, number_tries=1000):
    fname = args.problem

    if fname=='clique_v4':
        fname=fname+'_old'
        
    batch = data.batch
    num_graphs = batch.max().item() + 1
    edge_index= data.edge_index.detach()
    row, col = edge_index
    x = data.x 

    batch_data_list = data.to_data_list()
    random_set_values = torch.zeros(num_graphs)

    for counter, graph in enumerate(batch_data_list):
        best=np.inf
        for _ in range(number_tries):

            function_dict = {"function_name": fname, "graph": graph, "is_undirected": True, "cardinality_const": args.cardinality_const}
            if 'k_clique' in fname:
                l=torch.tensor(list(range(len(x))))
                idx=np.random.choice(l, size=args.k_clique_no, replace=False)
                random_set=torch.zeros_like(x)
                random_set[idx]=1
            else:
                random_set=(torch.rand(graph.x.shape[0])<args.rand_prob).float()
                #breakpoint()
            _, value=call_set_function(random_set, function_dict, args.penalty, args.k_clique_no)
            if value<=best:
                best=value

        random_set_values[counter]=best

    return random_set_values








