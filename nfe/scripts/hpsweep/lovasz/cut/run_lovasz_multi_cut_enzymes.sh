#!/bin/bash


 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 64 \
 				--depths 6 8 10 16 \
 				--l_rates 0.0001 \
 				--widths 64 128 256 \
 				--base_gnn gat \
                --features degree \
 				--problem cut \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz_multi \
                --n_sets 4


