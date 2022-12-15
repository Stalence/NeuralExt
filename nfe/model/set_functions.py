import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
import itertools
import torch.nn as nn
from extensions.extension_loader import get_extension_functions



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def call_set_function(input_set, function_dict, args, penalty=None, spanning_tr=False):
    function_name = function_dict['function_name']
    if (len(input_set.shape)==1) and not (function_name=="clique_v4_old"):
        input_set=input_set.unsqueeze(0)

    if "cut" in function_name:
        edge_index = function_dict["graph"].edge_index
        is_undir = function_dict["is_undirected"]
        cut=calculate_l1_cut(input_set, edge_index, undirected=is_undir)

        if  "min" in function_name:
            return  cut-args.max_val, cut
        else:
            return -cut, -cut

    elif function_name=="clique_v4":
       # breakpoint()
        cardinalities = input_set.sum(-1)
        #print(cardinalities)
        #breakpoint()

        mr, mc = function_dict["graph"].edge_index
        #breakpoint()


        weights =  (input_set.T[mr]*input_set.T[mc]).sum(0)*0.5
        #breakpoint()


        max_weights = (cardinalities*(cardinalities-1))*0.5 #max(cardinalities*(cardinalities-1)*0.5, 1)
        #breakpoint()
        max_weights = torch.max(max_weights, torch.tensor(1.))
        #breakpoint()

        edge_distance = ((weights/max_weights))*1.

        #breakpoint()

        is_clique = (edge_distance>=1)*1.
        #breakpoint()

        #breakpoint()

        loss=-(edge_distance*edge_distance)*weights 
        #breakpoint()
        is_clique[0]=1.

        #breakpoint()
        return loss, (-cardinalities * is_clique)


    elif function_name=="clique_4thpower":

        cardinalities = input_set.sum(-1)
        #print(cardinalities)
        mr, mc = function_dict["graph"].edge_index


        weights =  (input_set.T[mr]*input_set.T[mc]).sum(0)*0.5
        max_weights = (cardinalities*(cardinalities-1))*0.5 #max(cardinalities*(cardinalities-1)*0.5, 1)
        max_weights = torch.max(max_weights, torch.tensor(1.))
        edge_distance = ((weights/max_weights))*1.


        is_clique = (edge_distance>=1)*1.

        #breakpoint()

        loss=-(edge_distance*edge_distance*edge_distance*edge_distance)*weights 
        loss= loss*100.

        #is_clique[0]=1.

        #breakpoint()
        return loss, (-cardinalities * is_clique)

    elif function_name=="clique_v4_old":
        edge_distance = clique_distance(x=input_set, edge_index=function_dict["graph"].edge_index, undirected=function_dict["is_undirected"])
        node_distance = cardinality_distance(x=input_set, edge_index=function_dict["graph"].edge_index, undirected=function_dict["is_undirected"])
        
        is_clique = (edge_distance<=-1)*1.
        clique_size=input_set.sum()

        loss=(edge_distance*edge_distance)*node_distance
        return loss, (-clique_size * is_clique)


    elif function_name=="max_indep_set":
        edge_distance = mis_distance(x=input_set, edge_index=function_dict["graph"].edge_index, undirected=function_dict["is_undirected"])
        node_distance = mis_cardinality_distance(x=input_set, edge_index=function_dict["graph"].edge_index, undirected=function_dict["is_undirected"])
        
        is_mis_bool = (edge_distance==0)*1.
        mis_size=input_set.sum(dim=-1)


        loss=-(1-edge_distance)*(1-edge_distance)*node_distance

        #breakpoint()
        return loss, (-mis_size * is_mis_bool)
    

    elif function_name=="max_indep_set_RF":
        edge_distance = mis_distance(x=input_set, edge_index=function_dict["graph"].edge_index, undirected=function_dict["is_undirected"])
        node_distance = mis_cardinality_distance(x=input_set, edge_index=function_dict["graph"].edge_index, undirected=function_dict["is_undirected"])
        
        is_mis_bool = (edge_distance==0)*1.
        mis_size=input_set.sum(dim=-1)


        loss=-(1-edge_distance*edge_distance*edge_distance*edge_distance)*node_distance
        loss = loss*100.

        #breakpoint()
        return loss, (-mis_size * is_mis_bool)

    if function_name=="k_clique":

        cardinalities = input_set.sum(-1)
        #print(cardinalities)
        mr, mc = function_dict["graph"].edge_index


        weights =  (input_set.T[mr]*input_set.T[mc]).sum(0)*0.5
        max_weights = (cardinalities*(cardinalities-1))*0.5 #max(cardinalities*(cardinalities-1)*0.5, 1)
        max_weights = torch.max(max_weights, torch.tensor(1.))
        edge_distance = ((weights/max_weights))*1.


        is_clique = (edge_distance>=1)*1.

        loss=-(edge_distance*edge_distance)*weights 
     
        is_clique[0]=1.
        is_k_cardinality=(cardinalities>=args.k_clique_no)

        is_k_clique = is_clique * is_k_cardinality

        return loss, (-1 * is_k_clique)

    elif function_name=='coverage':
        graph=function_dict["graph"]
        input_set[graph.v]=0. #only look at subsets of u.
        cardinality_const=function_dict["cardinality_const"]

        selected_indices=list(input_set.nonzero().cpu().numpy().flatten())
        my_graph = to_networkx(Data(x=graph.x, edge_index = graph.edge_index)).to_undirected()

        neighbors_lst=[set(my_graph.neighbors(index)) for index in selected_indices]
        total_coverage=len(selected_indices)+len(set().union(*neighbors_lst)) #1sst term: number of nodes in U, 2nd term: number of nodes in V

        if penalty is None:
            pen=0.
        else:
            pen=penalty
        return -total_coverage + pen * max(0, len(selected_indices)-cardinality_const), (-total_coverage if len(selected_indices)<=cardinality_const else 0.)


#### HELPER FUNCTIONS FOR CUT PROBLEM #### 
def calculate_l1_cut(x, edge_index, undirected = True):
    #for binary vectors this gives the cut, 
    #for real vectors this gives the Lovasz extension of the cut.
    row, col = edge_index
    if undirected:
        cut = (torch.abs(x.T[row]-x.T[col])) * 0.5
    else:
        cut = (torch.abs(x.T[row]-x.T[col]))

    cut = cut.sum(0)
    return  cut  

#### HELPER FUNCTIONS FOR MIS PROBLEM #### 
def mis_cardinality_distance(x, edge_index, undirected):
    row, col = edge_index
    num_nodes = x.sum(dim=-1) 
    
    max_num_nodes = x.shape[0]

    return num_nodes/max_num_nodes

def mis_distance(x, edge_index, undirected):
    row, col = edge_index
    num_nodes = x.sum(dim=-1) 
    
    max_num_nodes = x.shape[0]
    max_num_edges= (num_nodes*(num_nodes-1))/2
    max_num_edges = torch.max(max_num_edges, torch.tensor(1.)) 

    if undirected:
        num_edges = (x.T[row]*x.T[col]).sum(dim=0)/2
        #num_edges = (x[row]*x[col]).sum()
    else:
        num_edges = (x.T[row]*x.T[col]).sum(dim=0)
      

    return num_edges/max_num_edges


#### HELPER FUNCTIONS FOR CLIQUE PROBLEM #### 
def clique_distance(x, edge_index, undirected):
    row, col = edge_index
    num_nodes = x.sum()
    
    max_num_nodes = x.shape[0]
    max_num_edges= (num_nodes*(num_nodes-1))/2
    
    if max_num_edges <= 1:
        return -1

    if undirected:
        num_edges = (x[row]*x[col]).sum()/2
    else:
        num_edges = (x[row]*x[col]).sum()
    
    distance = num_edges/max_num_edges
    assert(distance<=1)
    
    
    return -distance
    
def cardinality_distance(x, edge_index, undirected):
    row, col = edge_index
    num_nodes = x.sum()
    max_num_nodes = x.shape[0]
    max_num_edges = edge_index.shape[1]
    
    if undirected:
        num_edges = (x[row]*x[col]).sum()/2
    else:
        num_edges = (x[row]*x[col]).sum()

    edge_distance = num_edges/max_num_edges
    
    distance = num_nodes/max_num_nodes

    assert(edge_distance<=1)

    return -num_edges


