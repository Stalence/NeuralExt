#!/bin/bash

EIGS=8

python main.py --dataset_names ENZYMES \
 				--b_sizes 128 \
 				--depths 12 \
 				--l_rates 0.00001 \
 				--widths 256 \
 				--base_gnn gin \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension neural \
                --features lap_pe \
                --lap-method 'sign_inv' \
                --input_feat_dim 256 \
                --n_eigs $EIGS \
 				--one_dim_extension lovasz \
 			 	--neural v3 \
 			 	--n_sets 5 \
 				--debug \

