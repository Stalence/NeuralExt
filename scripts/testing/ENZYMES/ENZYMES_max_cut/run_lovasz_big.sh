#!/bin/bash

 python main.py --dataset_names ENZYMES \
 				--b_sizes 32 \
 				--depths 20 \
 				--l_rates 0.0001 \
 				--widths 512 \
 				--base_gnn gat \
 				--features degree \
 				--problem cut \
 				--rand_seeds 1 \
                --input_feat_dim 1 \
 				--epochs 600 \
 				--extension lovasz \
 				--debug