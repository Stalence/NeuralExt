#!/bin/bash

 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 \
 				--depths 8 \
 				--l_rates 0.0001 \
 				--widths 64 \
 				--base_gnn gat \
 				--features degree \
 				--problem k_clique \
 				--k_clique_no 3 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz \
 				--F1 \
 				--debug