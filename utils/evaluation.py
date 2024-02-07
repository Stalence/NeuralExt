from baselines import greedy_maximization, random_maximization, greedy_max
import torch
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def average_dataset_f1(net, loader, args, greedy=False, rand=False):
    TP, FP, TN, FN = 0, 0, 0, 0 

    with torch.no_grad():
        for data in loader:
            data=data.to(device)
            if greedy is True:
                #breakpoint()
                best_solutions=greedy_max(data, args).abs()
            elif rand is True:
                best_solutions=random_maximization(data, args).abs()
            else:
                output=net(data, args)
                best_solutions=torch.tensor(output['best_sets']).abs()

            predicted=best_solutions
            ground=data.ground.abs()


            pred_true=(predicted==1)
            tot_pred_true=pred_true.sum().item()

            TP_ = ground[pred_true].sum().item()
            FP_ = tot_pred_true - TP_

            pred_false=(predicted==0)
            tot_pred_false=pred_false.sum().item()

            FN_ = ground[pred_false].sum().item()
            TN_ = tot_pred_false - FN_

            TP+=TP_
            FP+=FP_
            TN+=TN_
            FN+=FN_

        F1 = TP / (TP + 0.5*(FP+FN))

        return torch.tensor(F1)

def single_item_ratio(best, ground, args):
    best=best.to(device)
    ground=ground.to(device)

    if args.problem=='k_clique':
        return (best==ground)
    elif args.problem=='min_cut':
        if ground==0:
            return (best+1)/(ground+1).item()
        else:
            return best/ground.item()
    else:
        return best/ground.item()

def average_dataset_ratio(net, loader, args, greedy=False, rand=False):
    ratios=[]

    with torch.no_grad():
        for data in loader:
            data=data.to(device)

            if greedy is True:
                greedy_solutions=greedy_max(data, args)
                ratios = ratios + [single_item_ratio(greedy_solutions[k], data.ground[k], args) for k in range(data.num_graphs)]
            elif rand is True:
                rand_solutions=random_maximization(data, args)

                ratios = ratios + [single_item_ratio(rand_solutions[k], data.ground[k], args) for k in range(data.num_graphs)]
            else:
                output=net(data, args)
                best_solutions=output['best_sets']
                ratios = ratios + [single_item_ratio(best_solutions[k], data.ground[k], args) for k in range(data.num_graphs)]

    ratios_std = torch.tensor(ratios).std()
    ratios = torch.tensor(ratios).mean()

    return ratios, ratios_std

def get_approx_ratio(net, loader, args):
    net.deterministic=True
    approx_ratio = torch.tensor(0.0, device=device)
    greedy_approx_ratio = torch.tensor(0.0, device=device)
    rand_approx_ratio = torch.tensor(0.0, device=device)
    net.eval()

    if args.F1 is True:
        approx_ratio=average_dataset_f1(net, loader, args)
        approx_ratio_std=torch.tensor(0.0, device=device)

    else:
        approx_ratio, approx_ratio_std = average_dataset_ratio(net, loader, args, greedy=False, rand=False)

    return (approx_ratio, approx_ratio_std)


def print_approx_ratio(net, train_loader, test_loader, val_loader, loss, time, epoch, args, train_approx_ratio=None):
    ######  This section computes greedy and random baselines 
    if (args.compute_greedy is True) or (args.compute_rand is True):
        if args.compute_greedy is True:
            if args.F1 is True:
                greedy_approx_ratio=average_dataset_f1(None, test_loader, args, greedy=True, rand=False)
            else:
                greedy_approx_ratio, _ =average_dataset_ratio(None, test_loader, args, greedy=True, rand=False)

            formatted_greedy_approx_ratio = "{:.3f}".format(greedy_approx_ratio.item())

            print(f'Epoch: {epoch} -- Greedy ratio: {formatted_greedy_approx_ratio}')
            


        if args.compute_rand is True:
            if args.F1 is True:
                rand_approx_ratio  =average_dataset_f1(None, test_loader, args, greedy=False, rand=True)
            else:
                rand_approx_ratio, _  =average_dataset_ratio(None, test_loader, args, greedy=False, rand=True)

            formatted_rand_approx_ratio = "{:.3f}".format(rand_approx_ratio.item())

            print(f'Epoch: {epoch} -- Rand ratio: {formatted_rand_approx_ratio}')


    else:
        if  train_approx_ratio is None:
            train_approx_ratio, _  = get_approx_ratio(net, train_loader, args)
        test_approx_ratio, test_approx_ratio_std   = get_approx_ratio(net, test_loader, args)
        val_approx_ratio, _    = get_approx_ratio(net, val_loader, args) 

        formatted_train_approx_ratio = "{:.3f}".format(train_approx_ratio.item())
        formatted_test_approx_ratio = "{:.3f}".format(test_approx_ratio.item())
        formatted_val_approx_ratio = "{:.3f}".format(val_approx_ratio.item())

        if epoch is None:
            print(f'Initialization -- Train ratio: {formatted_train_approx_ratio}, Test ratio: {formatted_test_approx_ratio}, Val ratio: {formatted_val_approx_ratio}')
        else:
            loss = "{:.3f}".format(loss)
            time = "{:.1f}".format(time)
            print(f'Epoch: {epoch} | Loss: {loss} | Time: {time} | Train ratio: {formatted_train_approx_ratio}, Test ratio: {formatted_test_approx_ratio}, Val ratio: {formatted_val_approx_ratio}')



        return test_approx_ratio, test_approx_ratio_std, val_approx_ratio

class StoreBestResults(object):
    """
    Keeps track of best val score, and the test score of the model with best val score.

    """
    def __init__(self, args):
        super(StoreBestResults, self).__init__()
        self.args=args
        self.best_val          = 0. if args.problem!='min_cut' else np.inf
        self.test_of_best_val = 0. if args.problem!='min_cut' else np.inf
        self.std=0.
        self.NonImprovementCounter=0

    def get_new_best(self, val_approx_ratio, test_approx_ratio, test_approx_ratio_std):
        if (val_approx_ratio>self.best_val) and (self.args.problem!='min_cut'):
            self.best_val=val_approx_ratio
            self.test_of_best_val=test_approx_ratio
            self.std=test_approx_ratio_std
            self.NonImprovementCounter=0


        elif (val_approx_ratio<self.best_val) and (self.args.problem=='min_cut'):
            self.best_val=val_approx_ratio
            self.test_of_best_val=test_approx_ratio
            self.std=test_approx_ratio_std
            self.NonImprovementCounter=0
        else:
            self.NonImprovementCounter+=1

        return self.best_val, self.test_of_best_val, self.std, self.NonImprovementCounter
