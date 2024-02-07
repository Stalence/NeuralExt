import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np

def preprocess_for_sampling(x, function_dict, args): 
    a = x @ x.T / np.sqrt(x.shape[0])

    adj=(torch_geometric.utils.to_dense_adj(function_dict["graph"].edge_index, max_num_nodes=a.shape[0])==1.).squeeze()
    for i in range(a.shape[0]):
        adj[i,i]=False

    a[~adj]=-np.inf #set score to -inf for all (i,j) pairs that are not edges. 
                    #after applyinig the softmax (below) this will give 0 transition prob to (i,j)

    for i in range(a.shape[0]):
        if (adj[i]==False).all():
            a[i]=-np.inf
            a[i,i]=1.

    transition_matrix=F.softmax(a, dim=1)

    return transition_matrix
        
def sample_set(i, sampling_data, args):
    a=sampling_data #n x n transition matrix

    n=a.shape[0]
    a_np=a.detach().cpu().numpy()

    j=np.random.choice(n)
    j_new=j

    terminate=False
    counter=0

    new_set, prob = {j}, 1./n
    while terminate is False:
        counter+=1

        j    = j_new
        j_new=np.random.choice(n, p=a_np[j])

        new_set.add(j_new)
        prob = prob * a[j,j_new]
        
        #if j_new in new_set:
        #    terminate=True

        if counter>=args.chain_length:
            terminate=True

    #print(len(new_set)) 
    level_set = torch.zeros(n)
    level_set[list(new_set)] = 1.

    return level_set, prob

