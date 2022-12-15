#!/bin/bash


 python main.py --dataset_names PROTEINS \
 				--problem max_indep_set \
 				--rand_seeds 1 \
                --compute-greedy \
                --debug
