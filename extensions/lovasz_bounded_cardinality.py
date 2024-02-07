import torch
import numpy as np
import torch.nn.functional as F

## Fast K-set implementation
def fast_kset_matrix(n,k):
    toe_column = torch.zeros(n,device='cuda')
    toe_row = torch.zeros(n,device='cuda')
    toe_row[0]=1
    toe_column[:k]=1
    toeplitz_mat = torch.zeros((n,n),device='cuda')
    tile_mat = toe_column*torch.ones_like(toe_column).unsqueeze(-1)
    tile_mat = tile_mat.T   
    for counter,row in enumerate(tile_mat):
        new_diag = torch.diag(row[counter:],diagonal=-counter)
        toeplitz_mat += new_diag
    return toeplitz_mat.T

## Fast Toe K inverse
def fast_toe_k_inverse(n, k):
    toe_column = torch.zeros(n,device='cuda')
    toe_row = torch.zeros(n,device='cuda')
    toe_row[0]=1
    toe_column[::k]=1
    where_one = torch.where(toe_column==1)[0]
    where_one = where_one[where_one<n-1]
    toe_column[where_one+1]=-1
    toeplitz_mat = torch.zeros((n,n),device='cuda')
    tile_mat = toe_column*torch.ones_like(toe_column).unsqueeze(-1)
    tile_mat = tile_mat.T   
    for counter,row in enumerate(tile_mat):
        new_diag = torch.diag(row[counter:],diagonal=-counter)
        toeplitz_mat += new_diag
    return toeplitz_mat.T




# def toe_k_inverse(n, k):
#     toe_column = torch.zeros(n,device='cuda')
#     toe_row = torch.zeros(n, device='cuda')
#     toe_row[0]=1
#     toe_column[::k]=1
#     where_one = torch.where(toe_column==1)[0]
#     where_one = where_one[where_one<n-1]
#     toe_column[where_one+1]=-1
#     return custom_toeplitz(toe_column,toe_row).T

# def k_set_matrix(n, k):
#     assert n>=k, "n must be larger than k!"
#     toe_column = torch.zeros(n,device='cuda')
#     toe_row = torch.zeros(n,device='cuda')
#     toe_row[0]=1
#     toe_column[:k]=1
#     return custom_toeplitz(toe_column,toe_row).T 

def preprocess_for_sampling(x, function_dict, args): 
    #x = x/x.norm()
    sorted_x, indices = x.sort(descending=True) 

    return x, sorted_x, indices

def sample_set(sampling_data, args):
    
    x, sorted_x, indices = sampling_data
    n = x.shape[0]
    k = args.bounded_k
    #x = x.softmax(dim=0)
    #scale by k helps with stability
    #x = k*x/x.norm(p=1)
    x= k*F.softmax(x)

    if k>=n:
        k=n

    #get starting triangular mat level sets
    b_level_sets = fast_kset_matrix(n,k)
    #get properly ordered level_sets
    permat = (indices*torch.ones_like(indices).unsqueeze(-1)).T
    level_sets = torch.zeros_like(b_level_sets).scatter_(0, permat,b_level_sets)
    
    #get probabilities
    probs = fast_toe_k_inverse(n,k)@sorted_x

    #probs*=100
    # #level_sets = get_level_sets(x, sorted_x)
    # level_sets = k_set_matrix(n,k)
    # level_sets =  level_sets[indices,:]
    #level_sets =  level_sets[:,flipped_indices_asc].T.flip(1)
    
    
    #This is transposed again during set func application so it's fine
    level_sets = level_sets.T
    #get probs by multiplying with 
    #breakpoint()
    #probs = toe_k_inverse(n,k).T@sorted_x
    #breakpoint()

    #probs = get_probs(sorted_x, args) #this is dumb, we are computing  all probs each time

    return level_sets, probs




def get_level_sets_v2(indices, x):
    sets=[]
    for i in range(len(indices)):
        level_set = torch.zeros(indices.shape[0])
        level_set[indices[:i+1]] = 1

        sets.append(level_set.unsqueeze(0))

    level_sets = torch.cat(sets, dim=0).to(x.device)
    #breakpoint()
    return level_sets
### LOVASZ UTILITY FUNCTIONS ###

def get_probs(sorted_x, args):
    sorted_x_roll =  sorted_x.roll(-1)
    sorted_x_roll[-1] = -args.max_val if args.problem=='min_cut' else  0.
    probs = sorted_x - sorted_x_roll
    #breakpoint()
    return probs