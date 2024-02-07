#!/bin/bash
#--l_rates 0.0001 \
 python main.py --dataset_names ENZYMES \
 				--b_sizes 8 \
 				--depths 10 \
 				--l_rates 0.0001 \
 				--widths 512 \
 				--base_gnn gat \
 				--features degree \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz \
 				--debug \
