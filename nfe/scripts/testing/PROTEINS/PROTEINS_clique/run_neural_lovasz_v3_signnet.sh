#!/bin/bash

EIGS=8

 python main.py --dataset_names PROTEINS \
 				--b_sizes 32 \
 				--depths 8 \
 				--l_rates 0.0001 \
 				--widths 64 \
 				--base_gnn gat \
                --features lap_pe \
                --lap-method 'sign_inv' \
                --input_feat_dim 64 \
                --n_eigs $EIGS \
                --problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension neural \
 				--one_dim_extension lovasz \
 			 	--neural v3 \
 			 	--n_sets 4 \
 				--debug \

