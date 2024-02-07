import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GatedGraphConv, GCNConv, SAGEConv, EGConv, GATConv



def get_sign_inv_net(hidden1, args):
    if args.lap_method=='sign_inv':
        SignNet = GINDeepSigns(1, hidden1, hidden1, gin_layers=2, mlp_layers=3, mlp_out_dim=hidden1, use_bn=True, dropout=False, activation='relu')
    else:
        SignNet=None

    return SignNet

class GINDeepSigns(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gin_layers, mlp_layers, mlp_out_dim, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):
        super(GINDeepSigns, self).__init__()

        convs = [GINConv(MLP(in_channels, hidden_channels, out_channels, mlp_layers, use_bn, use_ln, dropout, activation), train_eps=False)]
        convs = convs + [GINConv(MLP(hidden_channels, hidden_channels, out_channels, mlp_layers, use_bn, use_ln, dropout, activation), train_eps=False) for _ in range(gin_layers-1)]
        self.phi = nn.Sequential(*convs)
        self.rho = MLP(hidden_channels, hidden_channels, mlp_out_dim, mlp_layers, use_bn, use_ln, dropout, activation)

    def forward(self, x, edge_index):
        n_eigs=x.shape[1]
        embeddings=[]

        for i in range(n_eigs):
            x_i=x[:,i].unsqueeze(-1)

            x_i_plus = x_i
            x_i_minus = -x_i

            for conv in self.phi:
                x_i_plus  = conv(x_i_plus, edge_index)
                x_i_minus = conv(x_i_minus, edge_index)

            x_i = x_i_plus + x_i_minus
            x_i = x_i.unsqueeze(-1)

            embeddings.append(x_i)

        x = torch.cat(embeddings, dim=-1)
        x = x.mean(dim=-1)
        x  = self.rho(x)

        return x

        
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_bn=False, use_ln=False, dropout=0.5, activation='relu', residual=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual
            
    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2,1)).transpose(2,1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_bn=False, use_ln=False, dropout=0.5, activation='relu', residual=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual
            
    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2,1)).transpose(2,1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x
