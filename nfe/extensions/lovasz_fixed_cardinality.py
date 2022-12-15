import torch


def preprocess_for_sampling(x, function_dict, args): 

    #eps = (1e-3)*torch.rand(x.shape).to(x.device)
    #x = x + eps

    sorted_x, indices = x.sort(descending=True) 

    return x, sorted_x, indices

def sample_set(sampling_data, args):
    x, sorted_x, indices = sampling_data

    #level_sets = get_level_sets(x, sorted_x)
    level_sets = get_level_set(indices, x, args.k_clique_no)

    probs = get_probs(sorted_x, args) #this is dumb, we are computing  all probs each time

    return level_sets, probs


### LOVASZ UTILITY FUNCTIONS ###
def get_probs(sorted_x, args):
    #breakpoint()
    n=len(sorted_x)
    if args.k_clique_no>=n: 
        #handling case of graph of size less than k
        k=1
    else:
        k=args.k_clique_no

    """
    probs=[]
    for i in range(k-1,n):
        prob=sorted_x[i-k+1:i+1]
        probs.append(prob.unsqueeze(0))
    """
    yes_probs=[]
    no_probs=[]

    for i in range(k-1,n):
        yes=set(range(i-k+1,i+1))
        no=set(range(n)).difference(yes)
               
        yes=list(yes)
        no=list(no)

        yes_probs.append(sorted_x[yes].unsqueeze(0))
        no_probs.append(sorted_x[no].unsqueeze(0))


    #probs = [sorted_x[i-k+1:i+1].unsqueeze(0) for i in range(k-1,n)]

    yes_probs = torch.cat(yes_probs, dim=0)
    no_probs = torch.cat(no_probs, dim=0)

    probs = (yes_probs.log().sum(-1)+no_probs.log().sum(-1)).exp()

    #probs = torch.cat(probs, dim=0)
    #probs = probs.log().sum(-1).exp()

    return probs


### LOVASZ UTILITY FUNCTIONS ###
def get_probs_old(sorted_x, args):
    sorted_x_roll =  sorted_x.roll(-1)
    sorted_x_roll[-1] = -args.max_val if args.problem=='min_cut' else  0.
    probs = sorted_x - sorted_x_roll

    if args.k_clique_no>=len(sorted_x): 
        #handling case of graph of size less than k
        return probs
    else:
        return probs[args.k_clique_no-1:len(sorted_x)]

def get_level_set(indices, x, k):
   #breakpoint()
    if k>=len(indices): 
        #handling case of graph of size less than k
        k=1
    sets=[]
    for i in range(k-1,len(indices)):
        level_set = torch.zeros(indices.shape[0])
        level_set[indices[i-k+1:i+1]] = 1

        sets.append(level_set.unsqueeze(0))

    level_sets = torch.cat(sets, dim=0).to(x.device)
    #breakpoint()

    return level_sets


