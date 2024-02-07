#!/bin/bash

 python main.py --dataset_names COLLAB \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.00001 \
                --widths 256 \
 				--base_gnn gat \
 				--features one \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
                --erdos \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \


 python main.py --dataset_names COLLAB \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.00001 \
                --widths 256 \
                --base_gnn gat \
                --features one \
                --problem cut \
                --rand_seeds 1 \
                --epochs 200 \
                --erdos \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \


 python main.py --dataset_names COLLAB \
                --b_sizes 4 32 64 \
                --depths 6 10 16 \
                --l_rates 0.00001 \
                --widths 256 \
                --base_gnn gat \
                --features one \
                --problem max_indep_set \
                --rand_seeds 1 \
                --epochs 200 \
                --erdos \
                --dataset_scale 0.1 \
                --early_stop \
                --patience 30 \