#!/bin/bash

 python main.py --dataset_names PROTEINS \
 				--b_sizes 4  \
 				--depths 10 \
 				--l_rates 0.0001 \
 				--widths 128 \
 				--base_gnn gat \
 				--features degree \
 				--problem clique_v4 \
 				--rand_seeds 1 \
 				--epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 1 \


 python main.py --dataset_names PROTEINS \
                --b_sizes 4  \
                --depths 10 \
                --l_rates 0.0001 \
                --widths 128 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 2 \

 python main.py --dataset_names PROTEINS \
                --b_sizes 4  \
                --depths 10 \
                --l_rates 0.0001 \
                --widths 128 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 3 \

 python main.py --dataset_names PROTEINS \
                --b_sizes 4  \
                --depths 10 \
                --l_rates 0.0001 \
                --widths 128 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 4 \


 python main.py --dataset_names PROTEINS \
                --b_sizes 4  \
                --depths 10 \
                --l_rates 0.0001 \
                --widths 128 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 5 \

 python main.py --dataset_names PROTEINS \
                --b_sizes 4  \
                --depths 10 \
                --l_rates 0.0001 \
                --widths 128 \
                --base_gnn gat \
                --features degree \
                --problem clique_v4 \
                --rand_seeds 1 \
                --epochs 200 \
                --extension neural \
                --one_dim_extension lovasz \
                --neural v4 \
                --n_sets 6 \


