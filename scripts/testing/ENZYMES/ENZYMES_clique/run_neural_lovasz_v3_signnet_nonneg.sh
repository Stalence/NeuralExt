#!/bin/bash

EIGS=3

python main.py --dataset_names MUTAG \
 				--b_sizes 64 \
 				--depths 3 \
 				--l_rates 0.0001 \
 				--widths 16 \
 				--base_gnn gat \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension nonnegative \
                --features lap_pe \
                --lap-method 'sign_inv' \
                --input_feat_dim 16 \
                --n_eigs $EIGS \
 				--one_dim_extension lovasz \
 			 	--neural v3 \
 			 	--n_sets 4 \
 				--debug \
				--doubly_nonnegative

