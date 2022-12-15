import torch


def preprocess_for_sampling(x, function_dict, args): 

    #eps = (1e-3)*torch.rand(x.shape).to(x.device)
    #x = x + eps

    sorted_x, indices = x.sort(descending=True) 

    return x, sorted_x, indices

def sample_set(sampling_data, args):
    x, sorted_x, indices = sampling_data

    #level_sets = get_level_sets(x, sorted_x)
    level_sets = get_level_sets_v2(indices, x)

    probs = get_probs(sorted_x, args) #this is dumb, we are computing  all probs each time

    return level_sets, probs
        
def sample_set_old(i, sampling_data, args):
    _, sorted_vals, indices = sampling_data

    level_set = get_level_set_old(indices, i)
    probs = get_probs(sorted_vals, args) #this is dumb, we are computing  all probs each time

    return level_set, probs[i]


def sample_set_multi(sampling_data, args):
    x, sorted_x, indices = sampling_data

    #level_sets = get_level_sets(x, sorted_x)

    n = x.shape[0]

    level_sets=[]
    probs=[]

    for i in range(n):
        level_sets.append(get_level_sets_v2(indices[i], x[i]))
        probs.append(get_probs(sorted_x[i], args)) #this is dumb, we are computing  all probs each time

    level_sets = torch.cat(level_sets)
    probs      = torch.cat(probs)
    return level_sets, probs


### LOVASZ UTILITY FUNCTIONS ###
def get_probs(sorted_x, args):
    sorted_x_roll =  sorted_x.roll(-1)
    sorted_x_roll[-1] = -args.max_val if args.problem=='min_cut' else  0.
    probs = sorted_x - sorted_x_roll

    return probs


def get_level_sets(x, sorted_x):
    level_sets = ((x.unsqueeze(-1)>=(sorted_x.unsqueeze(-1)).T)*1.).T

    return level_sets

def get_level_sets_v2(indices, x):
    sets=[]
    for i in range(len(indices)):
        level_set = torch.zeros(indices.shape[0])
        level_set[indices[:i+1]] = 1

        sets.append(level_set.unsqueeze(0))

    level_sets = torch.cat(sets, dim=0).to(x.device)
    #breakpoint()

    return level_sets

def get_level_set_old(indices, i):
    level_set = torch.zeros(indices.shape[0])
    level_set[indices[:i+1]] = 1

    return level_set

