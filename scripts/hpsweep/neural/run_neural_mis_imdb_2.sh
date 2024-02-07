#!/bin/bash

 python main.py --dataset_names IMDB-BINARY\
 				--b_sizes 4 32 64 \
 				--depths 6 10 16 \
 				--l_rates 0.0001 \
 				--widths 64 \
 				--base_gnn gat \
 				--features degree \
 				--problem max_indep_set \
 				--rand_seeds 1 \
 				--epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 4 \


 python main.py --dataset_names IMDB-BINARY\
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.00001 \
                --widths 256 \
                --base_gnn gat \
                --features degree \
                --problem max_indep_set \
                --rand_seeds 1 \
                --epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 4 \


