#!/bin/bash

 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 \
 				--depths 20 \
 				--l_rates 0.0001 \
 				--widths 512 \
 				--base_gnn gat \
 				--features degree \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension neural \
 				--one_dim_extension lovasz \
 			 	--neural v3 \
 			 	--n_sets 4 \
 				--debug \

