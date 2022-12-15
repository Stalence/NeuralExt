import wandb
from datetime import date

class  Logger(object):
    def __init__(self, args, hparams):
        super(Logger, self).__init__()

        self.args=args

        if not args.debug:
            self.run = wandb.init(project="comb-opt" if args.local is True else "Lovasz_Extension_Project", entity='joshrob' if args.local is True else "epflmit", reinit=True, tags=[args.problem, "Josh"])
            config = wandb.config    

        curr_date = date.today()
        curr_date = curr_date.strftime("%B %d, %Y")
    
        batch_size, learning_rate, numlayers, r_seed, hidden_1, dataset_z = hparams 
        if not args.debug:
            config.batch_size = batch_size
            config.learning_rate = learning_rate
            config.curr_date = curr_date
            config.dataset = dataset_z[1]
            config.depth = numlayers 
            config.width = hidden_1
            config.seed  = r_seed
            config.problem = args.problem
            config.penalty = args.penalty
            config.features = args.features
            config.n_pertubations = args.n_pertubations
            config.curriculum = args.curriculum
            config.k_min = args.k_min
            config.k_max = args.k_max
            config.k_clique_no = args.k_clique_no
            config.experiment = args.experiment

            config.extension=args.extension
            config.one_dim_extension=args.one_dim_extension
            config.n_sets = args.n_sets

            config.lap_method= args.lap_method
            config.straight_through=args.straight_through
            config.reinforce=args.reinforce
            config.num_reinforce_samples=args.num_reinforce_samples
            config.erdos=args.erdos
            config.erdos_penalty=args.erdos_penalty
            config.straight_through_samples=args.straight_through_samples


    def log(self, info):
        if not self.args.debug:
            wandb.log(info)


    def finish(self):
        if not self.args.debug:
            self.run.finish()