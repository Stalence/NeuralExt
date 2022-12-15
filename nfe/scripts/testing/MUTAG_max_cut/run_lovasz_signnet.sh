#!/bin/bash


EIGS=8
#INPUT_DIM=16

python main.py --dataset_names MUTAG \
 				--b_sizes 32 \
 				--depths 8 \
 				--l_rates 0.0001 \
 				--widths 64 \
 				--base_gnn gat \
 				--problem cut \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz \
 				--features lap_pe \
 				--lap-method 'sign_inv' \
 				--input_feat_dim 64 \
 				--n_eigs $EIGS \
 				--debug