import extensions.lovasz as lovasz
import extensions.karger as karger  
import extensions.random_walk as random_walk
import extensions.lovasz_fixed_cardinality as lovasz_fixed_cardinality 
import extensions.lovasz_bounded_cardinality as lovasz_bounded_cardinality

import extensions.neural as neural
import extensions.nonnegative as nonnegative

lovasz_dic=lovasz.__dict__
karger_dic=karger.__dict__
random_walk_dic=random_walk.__dict__
lovasz_fixed_cardinality_dic=lovasz_fixed_cardinality.__dict__
lovasz_bounded_cardinality_dic = lovasz_bounded_cardinality.__dict__
neural_dic=neural.__dict__
nonnegative_dic = nonnegative.__dict__



def get_extension_functions(extension):
    if extension=='lovasz':
        preprocess_for_sampling = lovasz_dic['preprocess_for_sampling']
        sample_set = lovasz_dic['sample_set']

    elif extension=='lovasz_fixed_cardinality':
        preprocess_for_sampling = lovasz_fixed_cardinality_dic['preprocess_for_sampling']
        sample_set = lovasz_fixed_cardinality_dic['sample_set']

    elif extension=='lovasz_bounded_cardinality':
        preprocess_for_sampling = lovasz_bounded_cardinality_dic['preprocess_for_sampling']
        sample_set = lovasz_bounded_cardinality_dic['sample_set']

    elif extension=='lovasz_old':
        preprocess_for_sampling = lovasz_dic['preprocess_for_sampling']
        sample_set = lovasz_dic['sample_set_old']

    elif extension=='karger':
        preprocess_for_sampling = karger_dic['preprocess_for_sampling']
        sample_set = karger_dic['sample_set']

    elif extension=='random_walk':
        preprocess_for_sampling = random_walk_dic['preprocess_for_sampling']
        sample_set = random_walk_dic['sample_set']

    elif extension=='neural':
        preprocess_for_sampling = neural_dic['preprocess_for_sampling']
        sample_set = neural_dic['sample_set']

    elif extension=='nonnegative':
        preprocess_for_sampling = nonnegative_dic['preprocess_for_sampling']
        sample_set = nonnegative_dic['sample_set']

    elif extension=='lovasz_multi':
        preprocess_for_sampling = lovasz_dic['preprocess_for_sampling']
        sample_set = lovasz_dic['sample_set_multi']
    else:
        raise ValueError('Invalid extension name')


    return  preprocess_for_sampling, sample_set