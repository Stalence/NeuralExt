#!/bin/bash

 python main.py --dataset_names PROTEINS \
 				--b_sizes 4  \
 				--depths 10 \
 				--l_rates 0.0001 \
 				--widths 128 \
 				--base_gnn gat \
 				--features degree \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
                --extension lovasz \


 python main.py --dataset_names PROTEINS \
                --b_sizes 4  \
                --depths 10 \
                --l_rates 0.0001 \
                --widths 128 \
                --base_gnn gat \
                --features degree \
                --problem clique_4thpower \
                --rand_seeds 1 \
                --epochs 200 \
                --reinforce \
                --num_reinforce_samples 250 \




