#!/bin/bash

 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 \
 				--depths 8 \
 				--l_rates 0.0001 \
 				--widths 64 \
 				--base_gnn gat \
 				--features lap_pe \
 				--problem k_clique \
 				--k_clique_no 4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension neural \
 				--one_dim_extension lovasz \
 			 	--neural v3 \
 			 	--n_sets 4 \
 				--F1 \
 				--debug