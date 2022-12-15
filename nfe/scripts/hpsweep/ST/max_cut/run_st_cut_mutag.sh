#!/bin/bash

 python main.py --dataset_names MUTAG \
 				--b_sizes 4 32 64 \
 				--depths 6 10 16 \
 				--l_rates 0.0001 \
 				--widths 64 128 256 \
 				--base_gnn gat \
 				--features degree \
 				--problem cut \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--straight_through \
                --early_stop \
                --patience 30 \

