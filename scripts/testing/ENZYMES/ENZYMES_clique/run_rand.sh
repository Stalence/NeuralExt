#!/bin/bash
#--l_rates 0.0001 \
 python main.py --dataset_names ENZYMES \
 				--problem clique_v4 \
 				--rand_seeds 1 \
                --compute-rand \
                --rand-prob 0.1 \
                --debug
