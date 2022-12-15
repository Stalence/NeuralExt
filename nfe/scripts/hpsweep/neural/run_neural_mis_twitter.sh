#!/bin/bash

 python main.py --dataset_names TWITTER \
                --b_sizes 4 32 64 \
                --depths 6 10 \
                --l_rates 0.0001 \
                --widths 128 256 \
 				--base_gnn gat \
 				--features one \
                --problem max_indep_set \
                --rand_seeds 1 \
                --epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 4 \



