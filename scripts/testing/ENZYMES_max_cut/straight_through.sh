#!/bin/bash
python main.py --dataset_names ENZYMES \
			   --straight_through \
			   --num_reinforce_samples 1500 \ 
			   --depths 20 \
			   --widths 512  \
			   --b_sizes 4 \
			   --epochs 75 