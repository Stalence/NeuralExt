#!/bin/bash

EIGS=8

 python main.py --dataset_names IMDB-BINARY \
 				--b_sizes 32 \
 				--depths 8 \
 				--l_rates 0.0001 \
 				--widths 64 \
 				--base_gnn gat \
                --features lap_pe \
                --lap-method 'sign_inv' \
                --input_feat_dim 64 \
                --n_eigs $EIGS \
                --problem cut \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz \
 				--debug \
