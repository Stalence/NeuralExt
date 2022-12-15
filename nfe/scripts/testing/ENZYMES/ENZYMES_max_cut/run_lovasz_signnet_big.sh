#!/bin/bash

EIGS=4

 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 \
 				--depths 16 \
 				--l_rates 0.0001 \
 				--widths 512 \
 				--base_gnn gat \
                --features lap_pe \
                --lap-method 'sign_inv' \
                --n_eigs $EIGS \
 				--problem cut \
 				--rand_seeds 1 \
 				--epochs 600 \
 				--extension lovasz \
 				--debug