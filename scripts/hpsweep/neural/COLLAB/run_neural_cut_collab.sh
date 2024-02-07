#!/bin/bash

#--l_rates 0.0001 \
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
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 8 \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \


