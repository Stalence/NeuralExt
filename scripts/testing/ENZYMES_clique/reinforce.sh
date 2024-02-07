#!/bin/bash
 python main.py --dataset_names ENZYMES \
n
 				--base_gnn gat \
 				--features degree \
 				--problem clique_4thpower \
 				--rand_seeds 1 \
 				--epochs 200 \
 				--reinforce \
				--num_reinforce_samples 250 \