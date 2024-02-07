import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import numpy as np
from itertools import product
import pickle
import argparse
import timeit

#torch imports
import torch
import torch_geometric
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import autograd
from torch_geometric.utils import degree
from model.model import ExtensionNet
from baselines import greedy_maximization, random_maximization
from model.reinforce import ReinforceNet
from model.reinforce_with_baseline import ReinforceBaselineNet
from model.straight_through import STNet
from model.erdos import ErdosNet
from model.nonnegative_model import NonnegativeNet


from utils.evaluation import StoreBestResults, print_approx_ratio, single_item_ratio
import utils.log

import wandb
from data.data import get_datasets, get_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(net, train_loader, epoch, args):
    net.train()

    epoch_approx_ratio = torch.tensor(0.0, device=device)
    epoch_loss = 0.

    for count, data in enumerate(train_loader):
        optimizer.zero_grad(), 
        data = data.to(device)
        if args.aug is True:
            data.x=data.x + 0.1*torch.rand(data.x.shape).to(device)

        warmup=(args.warmup and epoch<10)
        output = net(data, args, warmup)

        loss = output["loss"]
        #print("loss grad:", loss.grad)
        #print("loss period", loss)
        #breakpoint()
        loss.backward()
        epoch_loss += loss.item()/len(train_loader)
        best_solutions = output["best_sets"] 

        if args.print_best:
            print('Solutions:')
            print(sum([int(b.item())==-1 for b in best_solutions])/len(best_solutions))
            print([int(b.item()) for b in best_solutions])
            print(list([int(g.item()) for g in data.ground]))

        
        ##check approx_ratio 
        ratios = [single_item_ratio(best_solutions[k], data.ground[k], args) for k in range(data.num_graphs)]
        epoch_approx_ratio += (sum(ratios)/len(ratios))/(len(train_loader))  

        #grad clipping, maybe not necessary we'll see
        torch.nn.utils.clip_grad_norm_(net.parameters(),4.)
        optimizer.step()

    return epoch_loss, ratios, epoch_approx_ratio, output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ExtensionNet trainer')

    #training params
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--lr_decay_step_size', default=40, type=int, help='how many epochs between lr steps')
    parser.add_argument('--lr_decay_factor', default=0.95, type=float, help='ratiioi by which lr is decreased')
    parser.add_argument('--lr_lower_bound', default=0.00005, type=float, help='lowest lr allowed')
    parser.add_argument('--epochs', default=300, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--b_sizes', default=[32], nargs='*', type=int, help='batch sizes to search over')
    parser.add_argument('--l_rates', default=[0.0001], nargs='*', type=float, help='learning rates to search over')
    parser.add_argument('--depths', default=[5], nargs='*', type=int, help='number of layers to search over')
    parser.add_argument('--widths', default=[128], nargs='*', type=int, help='model widths to search over')
    parser.add_argument('--output_sets', default=1, type=int, help='how many output sets to predict')
    parser.add_argument('--base_gnn', default='gcn', type=str, help='architecture of first module of network')
    parser.add_argument('--rand_seeds', default=[1], nargs='*', type=int, help='ranom seeds to run')
    parser.add_argument('--final_penalty', default=3., type=float, help='final beta penalty value')
    parser.add_argument('--input_feat_dim', default=None, type=int, help='final beta penalty value')
    parser.add_argument('--n_eigs', default=8, type=int, help='----')

    parser.add_argument('--problem', default='cut', type=str, help='which problem to study')
    parser.add_argument('--k-clique-no', default=None, type=int, help='which problem to study')

    parser.add_argument('--features', default='random', type=str, help='which node features to use')
    parser.add_argument('--lap-method', default=None, type=str, help='which node features to use')
    parser.add_argument('--time_limit', default=None, type=float, help='gurobi time limit')

    parser.add_argument('--n_pertubations', default=1, type=int, help='number of different level set families to produce')
    parser.add_argument('--reweight', action='store_true', help='will print out what value the solutions attain')
    parser.add_argument('--permute_method', default=None, type=int, help='number of different level set families to produce')
    parser.add_argument('--window', default=None, type=int, help='window size')
    parser.add_argument('--penalty', default=None, type=float, help='beta penalty value')

    parser.add_argument('--k', default=4, type=int, help='cardinality size for fixed cardinality Lovasz')

    parser.add_argument('--aug', action='store_true', help='use node feature data augmentation')
    parser.add_argument('--testing', action='store_true', help='use node feature data augmentation')

    parser.add_argument('--compute-greedy', action='store_true', help='greedy max estimation')
    parser.add_argument('--compute-rand', action='store_true', help='greedy max estimation')
    parser.add_argument('--rand-prob', default=0.5, type=float, help='greedy max estimation')
    parser.add_argument('--F1', action='store_true', help='will print out what value the solutions attain')

    parser.add_argument('--debug', action='store_true', help='do not run wandb logging')

    parser.add_argument('--cardinality_const', default=5, type=int, help='add a cardinality constraint')
    parser.add_argument('--n_tries', default=None, type=int, help='use a non-exact ground truth for when no efficient method to solve exists')

    parser.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use')

    # debugging
    parser.add_argument('--print_best', action='store_true', help='will print out what value the solutions attain')
    parser.add_argument('--test_freq', default=None, type=int, help='how many epochs per tets evaluation')
    parser.add_argument('--local',  action='store_true', help='which problem to study')
    parser.add_argument('--curriculum',  default=None, type=int, help='how many epochs per tets evaluation')

    parser.add_argument('--experiment', action='store_true', help='will print out what value the solutions attain')

    parser.add_argument('--extension',  default='lovasz', type=str, help='how many epochs per tets evaluation')
    parser.add_argument('--one_dim_extension',  default=None, type=str, help='how many epochs per tets evaluation')
    parser.add_argument('--k_min',  default=2, type=int, help='how many epochs per tets evaluation')
    parser.add_argument('--k_max',  default=4, type=int, help='how many epochs per tets evaluation')
    parser.add_argument('--num_sets',  default=None, type=int, help='how many epochs per tets evaluation')
    parser.add_argument('--new_diff',  action='store_true', help='which problem to study')

    parser.add_argument('--warmup',  action='store_true', help='which problem to study')
    parser.add_argument('--neural',  default='v1', type=str, help='for prototyping different versions of neural extension')
    parser.add_argument('--eig_sym',  action='store_true', help='which problem to study')

    parser.add_argument('--n_sets',  default=None, type=int, help='how many random spanning trees to use')
    parser.add_argument('--max_val',  default=5., type=float, help='how many random spanning trees to use')

    parser.add_argument('--chain_length',  default=6, type=int, help='how many random spanning trees to use')

    # TSP data parameterss
    parser.add_argument('--tsp_n_points', default=1000, type=int, help='dataset size (train, val anad test)')
    parser.add_argument('--tsp_max_size', default=20, type=int, help='largest size graph')
    parser.add_argument('--tsp_box_size', default=2., type=float, help='choose box points lie within')

    #datasets and saving
    parser.add_argument('--dataset_names', default=["ENZYMES"] , nargs='*', type=str, help='datasets to run over')
    parser.add_argument('--dataset_scale', default=1., type=float, help='proportion of dataset to use')
    parser.add_argument('--data_root', default='/data/scratch/joshrob/data', type=str, help='root directory where data is found')
    parser.add_argument('--save_root', default='/data/scratch/joshrob/comb-opt', type=str, help='root directory where results are saved')
    parser.add_argument('--save_name', default=None, type=str, help='save filename')
    parser.add_argument('--reinforce', action='store_true', help='Reinforce baseline')
    parser.add_argument('--num_reinforce_samples', default=200, type=int, help="Number of samples that reinforce is trained on in each epoch")
    parser.add_argument('--straight_through', action='store_true', help='Straighthrough baseline')
    parser.add_argument('--erdos', action ='store_true', help='Erdos baseline')
    parser.add_argument('--num_erdos_samples', default=1000, type=int)
    parser.add_argument('--erdos_penalty', default=1.5, type=float, help="Value of the penalty coefficient for Erdos")
    parser.add_argument('--straight_through_samples',action= 'store_true', help='Sample Sets for straighthrough instead of fixing')
    parser.add_argument('--num_st_samples', default = 1000, type=int, help= "number of straight through samples")
    parser.add_argument('--real_nfe', action='store_true', help= "straight through samples")
    parser.add_argument('--save_evals', action='store_true', help= "straight through samples")
    #parser.add_argument('--reinforce_breakpoint', action="store_true", help="breakpoint for debugging")
    parser.add_argument('--bounded_k', default=3, type=int, help="Value of k for bounded cardinality extension")
    parser.add_argument('--reinforce_with_baseline', action='store_true', help='Use Reinforce with baseline')
    parser.add_argument('--ER_scale_experiment', action='store_true', help = 'Erdos renyi scaling and time experiment')
    parser.add_argument('--ER_howmany', default=10, type=int, help='how many graphs per size')
    parser.add_argument('--ER_graph_size_ub', default=500, type=int, help='how many graphs per size')
    parser.add_argument('--ER_prob',default=0.75, type=float, help="parameter of ER model")
    parser.add_argument('--doubly_nonnegative', action = 'store_true', help= 'Completely positive matrix')
    

    parser.add_argument('--early_stop', action='store_true', help= "straight through samples")
    parser.add_argument('--patience', default=30, type=int, help="Value of k for bounded cardinality extension")

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    os.environ['WANDB_DIR']=args.save_root
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.save_evals_epoch=False

    if args.test_freq is None:
        args.test_freq=args.epochs+1 #i.e. never do test evaluation during training.

    datasets=get_datasets(args)

    manual_feat_dim=(args.input_feat_dim is None)
    
    for hparams in product(args.b_sizes, args.l_rates, args.depths, args.rand_seeds, args.widths, zip(datasets, args.dataset_names)):
        batch_size, learning_rate, numlayers, r_seed, hidden_1, dataset_z = hparams

        if manual_feat_dim:
            args.input_feat_dim=1

        #PREPARE DATA
        torch.manual_seed(r_seed)
        dataset = dataset_z[0]
        train_loader, test_loader, val_loader = get_loaders(dataset, batch_size)


        #network and loss stuff
        if args.reinforce:
            net = ReinforceNet(numlayers, hidden_1, args)
        elif args.straight_through:
             net = STNet(numlayers, hidden_1, args)
        elif args.reinforce_with_baseline:
             net = ReinforceBaselineNet(numlayers, hidden_1, args)
        elif args.erdos:
            net = ErdosNet(numlayers, hidden_1, args) 
        elif args.doubly_nonnegative:
            net = NonnegativeNet(numlayers, hidden_1, args)
        else:
            net = ExtensionNet(numlayers, hidden_1, args)
        net.to(device).reset_parameters()
        if args.optimizer=='adam':
            optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)
        if args.optimizer=='sgd':
            optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)


        net.train()

        print_approx_ratio(net, train_loader, test_loader, val_loader, None, None, None, args)
        if (args.compute_greedy is True) or (args.compute_rand is True):
            args.epochs=0 #no trainnig when computing baseline

        #with torch.autograd.set_detect_anomaly(True):
        logger=utils.log.Logger(args, hparams)
        best=StoreBestResults(args)

        

        for epoch in range(args.epochs):

            if (args.save_evals and epoch==20):
                args.save_evals_epoch=True
            else:
                args.save_evals_epoch=False

                

            net.deterministic=False

            start = timeit.default_timer()
            epoch_loss, ratios, epoch_approx_ratio, final_output = train_epoch(net, train_loader, epoch, args)
            stop = timeit.default_timer()

            test_approx_ratio, test_approx_ratio_std, val_approx_ratio = print_approx_ratio(net, train_loader, test_loader, val_loader, epoch_loss, start-stop, epoch, args, train_approx_ratio=epoch_approx_ratio)
            best_val, test_of_best_val, std, non_improvement = best.get_new_best(val_approx_ratio, test_approx_ratio, test_approx_ratio_std)

            log_info={  "loss": epoch_loss, \
                        "epoch": epoch, \
                        "train_approx_ratio": epoch_approx_ratio.item(), \
                        "test_approx_ratio": test_approx_ratio.item(), \
                        "best_val": best_val, \
                        "test_of_best_val":  test_of_best_val, \
                        "std":  std, \
                        "outputs": wandb.Histogram(final_output['x'].detach().cpu().numpy())}

            logger.log(log_info)

            if (non_improvement>args.patience) and (args.early_stop is True):
                break


        logger.finish()

