#!/bin/bash

 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 \
 				--depths 8 \
 				--l_rates 0.005 \
 				--widths 64 \
 				--base_gnn gat \
 				--features degree \
 				--problem min_cut \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension karger \
 				--n_sets 4 \
 				--debug \
 				#--dataset_scale 0.1
