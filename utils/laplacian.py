from baselines import greedy_maximization, random_maximization
import torch
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def handle_lap(data, SignNet=None, lap_method='sign_flip'):
    EigVals, EigVecs = data.e_vals, data.e_vecs
    if lap_method == 'sign_flip': 
        features=[]
        for counter, graph in enumerate(data.to_data_list()):
            val, vec = EigVals[data.batch==counter], EigVecs[data.batch==counter]  
            
            sign_flip = torch.rand(vec.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            vec = vec * sign_flip.unsqueeze(0)
            PosEnc = torch.cat((vec, val), dim=1).float()

            features.append(PosEnc)
        features=torch.cat(features)


    elif lap_method == 'sign_inv':
        edge_index = data.edge_index.detach()
        features = SignNet(EigVecs, edge_index)


    return features
