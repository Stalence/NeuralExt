#!/bin/bash

 python main.py --dataset_names COLLAB \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 256 \
 				--base_gnn gat \
 				--features one \
 				--problem clique_4thpower \
 				--rand_seeds 1 \
 				--epochs 200 \
                --reinforce \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \


 python main.py --dataset_names COLLAB \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 256 \
                --base_gnn gat \
                --features one \
                --problem cut \
                --rand_seeds 1 \
                --epochs 200 \
                --reinforce \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \


 python main.py --dataset_names COLLAB \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 256 \
                --base_gnn gat \
                --features one \
                --problem max_indep_set_RF \
                --rand_seeds 1 \
                --epochs 200 \
                --reinforce \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \