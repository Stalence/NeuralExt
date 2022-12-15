#!/bin/bash

 python main.py --dataset_names DD \
 				--b_sizes 4 32 64 \
 				--depths 6 10 16 \
 				--l_rates 0.0001 \
 				--widths 256 \
 				--base_gnn gat \
 				--features one \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--extension lovasz \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \

