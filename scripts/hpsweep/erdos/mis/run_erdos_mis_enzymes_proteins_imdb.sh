#!/bin/bash



 python main.py --dataset_names PROTEINS \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 64 128 256 \
                --base_gnn gat \
                --features degree \
                --problem max_indep_set \
                --rand_seeds 1 \
                --epochs 200 \
                --erdos

 python main.py --dataset_names IMDB-BINARY \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 64 128 256 \
                --base_gnn gat \
                --features degree \
                --problem max_indep_set \
                --rand_seeds 1 \
                --epochs 200 \
                --erdos





