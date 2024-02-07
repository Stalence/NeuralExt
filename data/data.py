import torch_geometric
from torch_geometric.datasets import TUDataset
from data.gurobi import get_ground_truth
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import degree, get_laplacian
import torch.nn.functional as F
from data.gurobi import solve_gurobi_maxclique
from data.gurobi import self_implement_degree
import networkx as nx
from torch_geometric.utils.convert import from_networkx

import pickle
import torch
import numpy as np 
import random
import os

def assign_features(graph,args):
    if args.problem=='tsp':
        args.input_feat_dim=2
    elif args.features=='random':
        if graph.x is not None:
            graph.x = torch.rand(graph.x.shape[0]) 
        else:
            graph.x = torch.rand(graph.num_nodes) 
    elif args.features=='one':
        if graph.x is not None:
            graph.x = torch.ones(graph.x.shape[0]) 
        else:
            graph.x = torch.ones(graph.num_nodes) 
        args.input_feat_dim=1
    elif args.features=='degree':
        args.input_feat_dim=1
        edge_index= graph.edge_index
        row, _ = edge_index
        #degs=self_implement_degree(row, graph.x.shape[0])
        degs = degree(row, num_nodes = graph.x.shape[0])
        x = (degs/degs.norm(1))*1.
        x = x.unsqueeze(-1)
        graph.x = x
    elif args.features=='lap_pe':
        n=graph.x.shape[0]

        a = get_laplacian(graph.edge_index)


        L = torch.zeros((n, n))

        row, col = a[0]

        L[row, col] = a[1]
        L[col, row] = a[1]
        # Eigenvectors with numpy
        EigVals, EigVecs = np.linalg.eigh(L.numpy())
        EigVals, EigVecs = EigVals[: args.n_eigs], EigVecs[:, : args.n_eigs]  # Keep up to the maximum desired number of frequencies

        # Normalize and pad EigenVectors
        EigVecs = torch.from_numpy(EigVecs).float()
        EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

        EigVals = torch.from_numpy(EigVals).float()
        EigVals = EigVals.repeat(n,1)


        if n<args.n_eigs:
            EigVals = F.pad(EigVals, (0, args.n_eigs-n), value=0.)
            EigVecs = F.pad(EigVecs, (0, args.n_eigs-n), value=0.)

        graph.e_vals, graph.e_vecs = EigVals, EigVecs


def get_baseline(graph,args):
    nx_graph = torch_geometric.utils.convert.to_networkx(graph).to_undirected()
    #breakpoint()
    if 'clique' in args.problem:
        baseline = len(nx.algorithms.approximation.max_clique(nx_graph))
    if 'indep_set' in args.problem:
        baseline = len(nx.algorithms.approximation.maximum_independent_set(nx_graph))
    if 'cut' in args.problem:
        baseline = nx.algorithms.approximation.one_exchange(nx_graph)[0]
    return baseline
    

def get_datasets(args):
    datasets=[]

    for name in args.dataset_names:

        if name=="TWITTER":
            stored_dataset = open('data/TWITTER_SNAP_2.p', 'rb')
            dataset = pickle.load(stored_dataset)
            #dataset=pickle.load(open(f"data/TWITTER_SNAP_2.p",'rb'))
            dataset=[Data(x=graph["x"], edge_index=graph["edge_index"]) for graph in list(dataset)]
        elif name=="ErdosRenyi":
            how_many_graph_sizes = args.ER_howmany
            graph_size_ub = args.ER_graph_size_ub
            graph_sizes = np.round(np.linspace(int(np.round(graph_size_ub/10)),graph_size_ub,how_many_graph_sizes))
            if os.path.exists(f'data/ground_truths/ErdosRenyi{how_many_graph_sizes}_{graph_size_ub}.p'):
                dictlist = pickle.load(open(f'data/ground_truths/ErdosRenyi{how_many_graph_sizes}_{graph_size_ub}.p','rb'))
                dataset = [Data.from_dict(graph) for graph in dictlist]
            else:
                num_graphs = 10
                graph_list = []
                pyg_graphs = []
                for counter,size in enumerate(graph_sizes):
                    for k in range(num_graphs):
                        print(f"Curr graph size : {counter+1}/{how_many_graph_sizes}, curr graph: {k+1}/{num_graphs}")
                        graph = nx.erdos_renyi_graph(int(size), p = args.ER_prob)
                        pyg_graph = from_networkx(graph)
                #pyg_graph.gurobi_scores = clique_sizes
                        pyg_graphs += [pyg_graph]

                dictlist = [graph.to_dict() for graph in pyg_graphs]        
                dataset= pyg_graphs
                pickle.dump(dictlist, open(f'data/ground_truths/ErdosRenyi{how_many_graph_sizes}_{graph_size_ub}.p','wb'))

        elif name=="BA":
            how_many_graph_sizes = args.ER_howmany
            graph_size_ub = args.ER_graph_size_ub
            graph_sizes = np.round(np.linspace(int(np.round(graph_size_ub/10)),graph_size_ub,how_many_graph_sizes))
            if os.path.exists(f'data/ground_truths/ErdosRenyi{how_many_graph_sizes}_{graph_size_ub}.p'):
                dictlist = pickle.load(open(f'data/ground_truths/ErdosRenyi{how_many_graph_sizes}_{graph_size_ub}.p','rb'))
                dataset = [Data.from_dict(graph) for graph in dictlist]
            else:
                num_graphs = 10
                graph_list = []
                pyg_graphs = []
                for counter,size in enumerate(graph_sizes):
                    for k in range(num_graphs):
                        print(f"Curr graph size : {counter+1}/{how_many_graph_sizes}, curr graph: {k+1}/{num_graphs}")
                        graph = nx.barabasi_albert_graph(int(size), int(size)/2)
                        pyg_graph = from_networkx(graph)
                #pyg_graph.gurobi_scores = clique_sizes
                        pyg_graphs += [pyg_graph]

                dictlist = [graph.to_dict() for graph in pyg_graphs]        
                dataset= pyg_graphs
                pickle.dump(dictlist, open(f'data/ground_truths/ErdosRenyi{how_many_graph_sizes}_{graph_size_ub}.p','wb'))
                
        else:
            dataset = TUDataset(root=f"./tmp/{name}", name=name)
            dataset = list(dataset)

        total_samples = int(np.floor(len(dataset)*args.dataset_scale))
        dataset = dataset[:total_samples]
        random.shuffle(dataset)

        #Get gurobi ground truths for maxcut  and clean the data a bit
        ground_truth_string = args.problem + name 
        if args.ER_scale_experiment:
            how_many_graph_sizes = args.ER_howmany
            graph_size_ub = args.ER_graph_size_ub
            ground_truth_string = ground_truth_string + '_' + str(how_many_graph_sizes) +'_' + str(graph_size_ub) + '_' +str(args.time_limit) + '_' + str(args.ER_prob)
        ground_truths = []
        print(f"Problem name and dataset: {ground_truth_string}")
        if os.path.exists(f'data/ground_truths/{ground_truth_string}.p'):
            ground_truths = pickle.load(open(f'data/ground_truths/{ground_truth_string}.p','rb'))
            print("Ground truths found! Skipping ground truth computation...")
        else:
             print('Ground truths not found! Computing ground truths...')
             for counter, graph in enumerate(dataset):
                if (counter+1) % 100 == 0:
                    print(f'Count: {counter+1}')
                graph_info = {}    
                num_nodes = graph.num_nodes
                num_edges = graph.num_edges
                ground = get_ground_truth(graph, args)
                graph_info['num_nodes'] = num_nodes
                graph_info['num_edges'] = num_edges
                graph_info['ground_truth'] = ground
                ground_truths += [graph_info]
             pickle.dump(ground_truths, open(f"data/ground_truths/{ground_truth_string}.p",'wb'))
             print("Ground truths saved.")

        #Assigning node features
        print("Assigning node features...")
        connected_idx=[]
        for counter, graph in enumerate(dataset):

            if graph.x is None:
                #breakpoint()
                graph.x=torch.zeros(graph.edge_index.max().item()+1) 
            assert graph.x.shape[0] > graph.edge_index.max().item()

            assign_features(graph, args)
            if args.reinforce_with_baseline:
                graph.baseline = get_baseline(graph, args)
               # breakpoint()

            if (counter+1) % 100 == 0:
                print(f'Count: {counter+1}')

            #ground = get_ground_truth(graph, args)
            #Match stored ground truths with our data and check if everything went well
            found_ground = False
            for graph_key in ground_truths:
                if ((graph.num_nodes==graph_key['num_nodes']) and (graph.num_edges==graph_key['num_edges'])):
                    graph.ground = graph_key['ground_truth']
                    found_ground = True
            if not found_ground:
                "WARNING! NO GROUND TRUTH MATCH FOUND"
                breakpoint()

            if graph.ground>0:
                connected_idx.append(counter)

            if args.print_best:
                print(graph.ground)

        if args.extension=='karger':
            dataset=[dataset[idx] for idx in connected_idx]
            print(f'{100*(1-(len(connected_idx)/len(dataset)))}% of graphs removed due to not being connected.')

        print('Done!')
        datasets.append(dataset)
        #breakpoint()
        return datasets

def get_loaders(dataset, batch_size):
    num_trainpoints = int(np.floor(0.6*len(dataset)))
    num_valpoints = int(np.floor(0.3*num_trainpoints))
    num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)

    traindata= dataset[0:num_trainpoints]
    valdata = dataset[num_trainpoints:num_trainpoints + num_valpoints]
    testdata = dataset[num_trainpoints + num_valpoints:]

    train_loader = DataLoader(traindata, batch_size, shuffle=True)
    test_loader  = DataLoader(testdata,  batch_size, shuffle=False)
    val_loader   = DataLoader(valdata,   batch_size, shuffle=False)

    return train_loader, test_loader, val_loader  


