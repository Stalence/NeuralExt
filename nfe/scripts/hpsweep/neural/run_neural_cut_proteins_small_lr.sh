#!/bin/bash

 python main.py --dataset_names PROTEINS \
 				--b_sizes 4 32 64 \
 				--depths 16 \
 				--l_rates 0.000001 \
 				--widths 256 \
 				--base_gnn gat \
 				--features degree \
 				--problem cut \
 				--rand_seeds 1 \
 				--epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 4 \


