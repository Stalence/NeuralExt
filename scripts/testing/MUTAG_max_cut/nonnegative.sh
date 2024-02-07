#!/bin/bash
python main.py --dataset_names MUTAG \
--b_sizes 32 \
--depths 8 \
--l_rates 0.0001 \
--widths 64 \
--base_gnn gat \
--features lap_pe \
--lap-method 'sign_inv' \
--n_eigs 8  \
--rand_seeds 1 \
--epochs 200 \
--extension neural \
--neural v3 \
--one_dim_extension lovasz \
--problem clique_v4 \
--n_sets 6 \
--debug