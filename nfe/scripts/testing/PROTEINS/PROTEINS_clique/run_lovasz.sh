#!/bin/bash

 python main.py --dataset_names PROTEINS \
 				--b_sizes 32 \
 				--depths 8 \
 				--l_rates 0.0001 \
 				--widths 64 \
 				--base_gnn gat \
 				--features degree \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz \
 				--debug \
