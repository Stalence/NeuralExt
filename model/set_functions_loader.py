import torch

from model.set_functions import call_set_function
from extensions.extension_loader import get_extension_functions

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_set_functions(extension, args):
    #handles high dimensional extension case
    if (extension=='neural') or (extension=="nonnegative"):
        def set_funct(x, function_dict, args, penalty=None, spanning_tr=False):
            one_dim_preprocess, one_dim_sample = get_extension_functions(args.one_dim_extension)
            one_dim_call_set_function = get_set_functions(args.one_dim_extension, args)

            sampling_data = one_dim_preprocess(x, function_dict, args)

            if args.one_dim_extension=='lovasz_old':
                n_sets = len(x) if args.one_dim_extension=='lovasz_old' else args.n_sets

                sets=n_sets*[0.]
                probs=n_sets*[0.]
                f_sets=torch.zeros(n_sets).to(device)
                f_unreg_sets=torch.zeros(n_sets).to(device)

                for i in range(n_sets):
                    sets[i], probs[i] = one_dim_sample(i, sampling_data, args)
                    f_sets[i], f_unreg_sets[i] = one_dim_call_set_function(sets[i], function_dict, args, penalty)


                probs=torch.cat([p.unsqueeze(0) for p in probs])

            elif args.one_dim_extension=='lovasz':
                sets, probs = one_dim_sample(sampling_data, args)
                f_sets, f_unreg_sets = one_dim_call_set_function(sets, function_dict, args, penalty)


            if args.neural=='v4':
                diag_term  = probs#**2
                flipped_probs=torch.cumsum(probs.flip(0), dim=0)-probs.flip(0)
                cross_term = 2 * flipped_probs * probs
                probs = diag_term + cross_term

            extension   = (f_sets*probs).sum()

            min_set_val = f_unreg_sets.min()

            return extension, min_set_val

    else:
        set_funct = call_set_function

    return  set_funct