#!/bin/bash

 python main.py --dataset_names ENZYMES \
 				--b_sizes 4 \
 				--depths 6 10 16 \
 				--l_rates 0.0001 \
 				--widths 64 128 256 \
 				--base_gnn gat \
 				--features degree \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--straight_through \


 python main.py --dataset_names PROTEINS \
                --b_sizes 4 \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 64 128 256 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --straight_through \


 python main.py --dataset_names IMDB-BINARY \
                --b_sizes 4  \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 64 128 256 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --straight_through \


 python main.py --dataset_names MUTAG \
                --b_sizes 4  \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 64 128 256 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --straight_through \


 python main.py --dataset_names AIDS \
                --b_sizes 4  \
                --depths 6 10 16 \
                --l_rates 0.0001 \
                --widths 64 128 256 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --straight_through \

