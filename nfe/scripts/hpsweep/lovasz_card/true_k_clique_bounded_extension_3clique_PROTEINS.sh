#!/bin/bash
python main.py  --dataset_names PROTEINS  \
                --b_sizes 4 32 \
                --depths 8 16 \
                --l_rates 0.0001 0.0000001  \
                --widths 64 256 \
                --base_gnn gat  \
                --features degree  \
                --problem k_clique  \
                --k-clique-no 3  \
                --rand_seeds 1  \
                --epochs 200  \
                --extension lovasz_bounded_cardinality \
                --bounded_k 3 \
                --F1  \

