#!/bin/bash

EIGS=8

 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 \
                --depths 6 8 10 \
                --l_rates 0.0001 0.005 0.001 \
                --widths 64 128 \
 				--base_gnn gat \
                --features lap_pe \
                --lap-method 'sign_inv' \
                --n_eigs $EIGS \
                --problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz \
                --save_root ~/nfe \
                --debug \

