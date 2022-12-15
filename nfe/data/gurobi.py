import gurobipy as gp
from gurobipy import GRB
import torch
import networkx as nx
import torch.nn.functional as F
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree, get_laplacian
#from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from torch_geometric.data import Data
from model.set_functions import call_set_function
import random
import numpy as np
import itertools
import math

import sys
from itertools import combinations

def self_implement_degree(lst, mx):
    #mx=max(lst)+1
    degs=torch.zeros(mx)
    for idx in range(mx):
        degs[idx]=sum(list(map(int, [j==idx for j in lst])))
        
    return degs


def get_ground_truth(graph, args):
    ###assign features to node
    ###compute ground truth solution values
    if args.problem=='cut':
        print('next')
        ground = -gurobi_maxcut(graph, time_limit=1.)

    elif args.problem=='min_cut':
        ground = gurobi_mincut(graph)

    elif args.problem=='coverage':
        xs=list(range(graph.x.shape[0])) 
        random.shuffle(xs)
        graph.u=xs[:len(xs)//2]
        graph.v=xs[len(xs)//2:]

        n_edges=graph.edge_index.shape[1]
        edges_to_keep=[]
        for e in range(n_edges):
            i, j = graph.edge_index[0,e].item(), graph.edge_index[1,e].item()

            if ((i in graph.u) and (j in graph.v)) or ((j in graph.u) and (i in graph.v)):
                edges_to_keep.append((i,j))
            else:
                pass

        ins, outs = zip(*edges_to_keep)
        graph.edge_index=torch.tensor([ins, outs])

        ground, _ = gurobi_maxcover(graph, args.cardinality_const)
        ground = -ground

    elif args.problem  in ["clique_v4", "clique_v4_old","clique_4thpower"]:
        my_graph = to_networkx(Data(x=graph.x, edge_index = graph.edge_index)).to_undirected()
        cliqno, _ = solve_gurobi_maxclique(my_graph, args.time_limit)
        ground = -cliqno


    elif args.problem=="k_clique":
        my_graph = to_networkx(Data(x=graph.x, edge_index = graph.edge_index)).to_undirected()
        cliqno, _ = solve_gurobi_maxclique(my_graph, 100000)
        ground = -(cliqno>=args.k_clique_no)

    elif args.problem=='shortest_path':
        ground = shortest_path(graph)

    elif args.problem in ['max_indep_set', 'max_indep_set_RF']:
        my_graph = to_networkx(Data(x=graph.x, edge_index = graph.edge_index)).to_undirected()
        mis = solve_gurobi_mis(my_graph, 100000)[0]

        ground = -mis

    elif args.problem=='tsp':
        ground=solve_gurobi_tsp([tuple(point) for point in graph.x])

    else:
        raise ValueError('Invalid problem name')



    return ground


def gurobi_maxcover(data, cardinality_bound, time_limit=None):
    x_vars = {}
    model = gp.Model("mip1")
    model.params.OutputFlag=0

    degrees = self_implement_degree(data.edge_index[0,:], data.x.shape[0])[data.u]

    if time_limit:
        model.params.TimeLimit = time_limit
        
    for node in data.u:
        x_vars['x_'+str(node)] = model.addVar(vtype=GRB.BINARY, name="x_"+str(node))


    model.addConstr(sum([x_vars['x_'+str(node)] for node in data.u]) <= cardinality_bound, 'cardinality_constraint')    
    model.setObjective(sum([x_vars['x_'+str(node)]*degrees[counter] for counter,node in enumerate(data.u)]), GRB.MAXIMIZE);

    # Optimize model
    model.optimize();

    coverage = model.objVal+cardinality_bound;
    x_vals = [var.x for var in model.getVars()] 

    return coverage, x_vals

def shortest_path(graph):
    graph = to_networkx(Data(edge_index = graph.edge_index))
    length = dict(nx.all_pairs_shortest_path_length(graph))
    return length

###define ground truth Gurobi solver
def gurobi_maxcut(graph, time_limit = None):
    # Create a new model

    model = gp.Model("maxcut")
    model.params.OutputFlag = 0

    # time limit in seconds, if applicable
    if time_limit:
        model.params.TimeLimit = time_limit
    
    # Create variables
    x_vars = {}
    for i in range(graph.num_nodes):
        x_vars["x_" + str(i)] = model.addVar(vtype=GRB.BINARY, name="x_" + str(i))

    # Set objective
    obj = gp.QuadExpr()
    for source, target in zip(*graph.edge_index.tolist()):
        qi_qj = (x_vars['x_' + str(source)] - x_vars['x_' + str(target)])
        obj += qi_qj * qi_qj / 2
    model.setObjective(obj, GRB.MAXIMIZE)

    # Optimize model
    model.optimize()
    return model.objVal

###define ground truth Gurobi solver
def gurobi_mincut(graph, time_limit = None):
    # Create a new model
    model = gp.Model("mincut")
    model.params.OutputFlag = 0

    # time limit in seconds, if applicable
    if time_limit:
        model.params.TimeLimit = time_limit
    
    # Create variables
    x_vars = {}
    for i in range(graph.num_nodes):
        x_vars["x_" + str(i)] = model.addVar(vtype=GRB.BINARY, name="x_" + str(i))

    # Set objective
    obj = gp.QuadExpr()
    for source, target in zip(*graph.edge_index.tolist()):
        qi_qj = (x_vars['x_' + str(source)] - x_vars['x_' + str(target)])
        obj += qi_qj * qi_qj / 2

    constraint_1 = model.addConstr(sum(v for v in x_vars.values()) >= 1)
    constraint_2 = model.addConstr(sum(v for v in x_vars.values()) <= len(x_vars)-1)
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize model
    model.optimize()
    
    return model.objVal

def solve_gurobi_maxclique(nx_graph, time_limit = None):
    nx_complement = nx.operators.complement(nx_graph)
    x_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag=0

    if time_limit:
        m.params.TimeLimit = time_limit

    for node in nx_complement.nodes():
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))

    count_edges = 0
    for edge in nx_complement.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1,'c_'+str(count_edges))
        count_edges+=1
    m.setObjective(sum([x_vars['x_'+str(node)] for node in nx_complement.nodes()]), GRB.MAXIMIZE);

    # Optimize model
    m.optimize();

    set_size = m.objVal;
    x_vals = [var.x for var in m.getVars()] 

    return set_size, x_vals


def solve_gurobi_mis(nx_graph, time_limit = None):

    x_vars = {}
    c_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag = 0

    if time_limit:
        m.params.TimeLimit = time_limit

    for node in nx_graph.nodes():
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))

    count_edges = 0
    for edge in nx_graph.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1,'c_'+str(count_edges))
        count_edges+=1
    m.setObjective(sum([x_vars['x_'+str(node)] for node in nx_graph.nodes()]), GRB.MAXIMIZE);


    # Optimize model
    m.optimize();

    set_size = m.objVal;
    x_vals = [var.x for var in m.getVars()] 

    return set_size, x_vals



###define ground truth Gurobi solver
def nx_max_indep_set(graph, time_limit = None):
    best_mis=0


    #mis=nx.maximal_independent_set(graph)  
    #mis=len(mis)
    #graph = nx.line_graph(graph)

    for node in graph:
        mis=nx.maximal_independent_set(graph, [node]) 
        mis=len(mis)
 
        best_mis=max(best_mis, mis)


    return best_mis


def solve_gurobi_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n=len(points)

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            # find the shortest cycle in the selected edge list
            tour = subtour(vals)
            if len(tour) < n:
                # add subtour elimination constr. for every pair of cities in tour
                model.cbLazy(gp.quicksum(model._vars[i, j]
                                         for i, j in combinations(tour, 2))
                             <= len(tour)-1)


    # Given a tuplelist of edges, find the shortest subtour

    def subtour(vals):
        # make a list of edges selected in the solution
        edges = gp.tuplelist((i, j) for i, j in vals.keys()
                             if vals[i, j] > 0.5)
        unvisited = list(range(n))
        cycle = range(n+1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle


    dist = {(i, j):
            math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            for i in range(n) for j in range(i)}

    m = gp.Model()
    m.params.OutputFlag=0

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction

    # Add degree-2 constraint
    m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))


    # Optimize model

    m._vars = vars
    m.Params.LazyConstraints = 1
    m.optimize(subtourelim)

    vals = m.getAttr('X', vars)
    tour = subtour(vals)
    assert len(tour) == n

    return m.ObjVal




"""
def solve_gurobi_tsp(points, threads=0, timeout=None, gap=None):

    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 


    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = gp.Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction


    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)

    breakpoint()
    assert len(tour) == n

    return m.objVal, tour
"""


def maxcover_random(graph, args):
    best=np.inf
    function_dict = {"function_name": args.problem, "graph": graph, "is_undirected": True, "cardinality_const": args.cardinality_const}
    
    if args.n_tries is not None:
        if args.cardinality_const>len(graph.u):
            combs=[graph.u]
        else:
            combs=[random.sample(graph.u, args.cardinality_const) for _ in range(args.n_tries)]
        for indices in combs:
            candidate_set=torch.zeros(graph.x.shape[0])
            candidate_set[indices]=1.
            _, value =call_set_function(candidate_set, function_dict, args.penalty)
            if value<best:
                best=value


    else:
        for indices in itertools.combinations(graph.u, args.cardinality_const): #only check sets of max card since max-cover an inctreasing function
            candidate_set=torch.zeros(graph.x.shape[0])
            candidate_set[list(indices)]=1.
            _, value =call_set_function(candidate_set, function_dict, args.penalty)
            if value<best:
                best=value

    ground=best

    return ground