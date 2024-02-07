#!/bin/bash


 python main.py --dataset_names PROTEINS \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 64 128 256 \
                --base_gnn gat \
                --features degree \
                --problem cut \
                --rand_seeds 1 \
                --epochs 200 \
                --reinforce \
                --num_reinforce_samples 250 \
                --early_stop \
                --patience 30 \

