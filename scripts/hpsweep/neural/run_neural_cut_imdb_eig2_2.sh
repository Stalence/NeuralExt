#!/bin/bash

 python main.py --dataset_names IMDB-BINARY\
 				--b_sizes 4 32 64 \
 				--depths 6 10 16 \
 				--l_rates 0.0001 \
 				--widths 128 \
 				--base_gnn gat \
 				--features degree \
 				--problem cut \
 				--rand_seeds 1 \
 				--epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 2 \
                --early_stop \
                --patience 30 \


