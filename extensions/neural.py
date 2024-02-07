import torch
import torch.nn.functional as F

import numpy as np

import scipy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_for_sampling(x, function_dict, args): 
    if args.neural in ['v3', 'v4']:
        #x = (x/x.norm(dim=1).unsqueeze(-1))
        #breakpoint()
        gram = x @ x.T 
        
    
        eigenvalues=[]
        eigenvectors=[]
        for _ in range(args.n_sets):
            gram, eigval, eigvec = power_method(gram)

            eigenvalues.append(eigval)
            eigenvectors.append(eigvec)
        
        eigenvalues = torch.cat(eigenvalues, dim=0)
        eigenvectors = torch.cat(eigenvectors, dim=1)
        #breakpoint()


        eigenvalues = F.softmax(eigenvalues, dim=0)


        pos_eigenvectors = eigenvectors #F.sigmoid(eigenvectors)


        if args.eig_sym is True:
            neg_eigenvectors = F.sigmoid(-eigenvectors)
            eigenvalues  = torch.cat([eigenvalues, eigenvalues], dim=0)
            all_eigenvectors = torch.cat([pos_eigenvectors, neg_eigenvectors], dim=0)

        else:
            all_eigenvectors=pos_eigenvectors
        
        breakpoint()
        return eigenvalues, all_eigenvectors

def preprocess_for_sampling_old(x, function_dict, args): 
    if args.neural in ['v3', 'v4']:
        gram = x @ x.T 

        eigenvalues=[]
        eigenvectors=[]
        for _ in range(args.n_sets):
            gram, eigval, eigvec = power_method(gram)

            eigenvalues.append(eigval)
            eigenvectors.append(eigvec)

        eigenvalues = torch.cat(eigenvalues, dim=0)
        eigenvectors = torch.cat(eigenvectors, dim=1)

        #normalize **after** computing eigendecomposition
        eigenvalues = F.softmax(eigenvalues, dim=0)

        #n=eigenvalues.shape[0]
        #eigenvalues = torch.ones(n).to(device)/n
        pos_eigenvectors = F.sigmoid(eigenvectors)


        if args.eig_sym is True:
            neg_eigenvectors = F.sigmoid(-eigenvectors)
            eigenvalues  = torch.cat([eigenvalues, eigenvalues], dim=0)
            all_eigenvectors = torch.cat([pos_eigenvectors, neg_eigenvectors], dim=0)

        else:
            all_eigenvectors=pos_eigenvectors
            
        return eigenvalues, all_eigenvectors


        
def sample_set(i, sampling_data, args):
    #breakpoint()
    eigenvalues, eigenvectors = sampling_data

    prob, x = eigenvalues[i], eigenvectors[:,i]

    return x, prob


##### utility functions #####
def warmup(x):
    n = x.shape[0]
        
    x = F.normalize(x, dim=1) / np.sqrt(n)
    X = x @ x.T 

    eigenvalues, _ = torch.linalg.eigh(X)

    uniform = torch.ones_like(eigenvalues).to(device) / n
    loss = F.mse_loss(eigenvalues, uniform)

    return loss


def power_method(gram):
    n=gram.shape[0]

    y = torch.rand(n, 1).to(device)
    y = y/y.norm()

    iterations=5

    for _ in range(iterations):
        y = F.normalize(gram @ y, dim=0)

    eigvec = y
    eigval = y.T @ gram @ y
    gram = gram - (eigval * (eigvec*eigvec.T))

    #old version
    #gram = gram - (eigval * (eigvec))

    return gram, eigval, eigvec



"""

    if args.neural=='v1':
        #add noise to make eigenvector backprop numerically stable
        x = x + torch.normal(mean=torch.zeros_like(x), std=1e-4 * torch.ones_like(x))
        X = x @ x.T 

        #eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(X)

        #normalize **after** computing eigendecomposition
        eigenvalues = F.softmax(eigenvalues, dim=0)
        eigenvectors = F.softmax(eigenvectors, dim=1)

        return eigenvalues, eigenvectors
    elif args.neural=='v2':
        #add noise to make eigenvector backprop numerically stable
        x = x + torch.normal(mean=torch.zeros_like(x), std=1e-4 * torch.ones_like(x))
        X = x @ x.T 

        #eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(X)

        #normalize **after** computing eigendecomposition
        eigenvalues = F.softmax(eigenvalues, dim=0)
        eigenvectors = F.sigmoid(eigenvectors)

        return eigenvalues, eigenvectors


def preprocess_for_sampling_test2(x, function_dict, args): 
    n = x.shape[0]

    x = x + torch.normal(mean=torch.zeros_like(x), std=0.0001 * torch.ones_like(x))
    #x=x.unsqueeze(dim=1) 
    #x = F.normalize(x, dim=1) / np.sqrt(n)

    X = x @ x.T 

    #X = X + (0.01 * torch.eye(n).to(device))

    eigenvalues, eigenvectors = torch.linalg.eigh(X)
    #breakpoint()
    #eigenvalues = eigenvalues / sum(eigenvalues)
    eigenvalues = F.softmax(eigenvalues, dim=0)

    #print(eigenvalues)
    #print(scipy.stats.entropy(eigenvalues.detach().cpu().numpy()))
    return eigenvalues, eigenvectors

def preprocess_for_sampling_test(x, function_dict, args): 
    n = x.shape[0]
    

    X = x @ x.T 

    eigenvalues, eigenvectors = torch.linalg.eigh(X)

    eigenvalues = F.softmax(eigenvalues)
    #print(eigenvalues)
    #print(scipy.stats.entropy(eigenvalues.detach().cpu().numpy()))
    return eigenvalues, eigenvectors

"""


