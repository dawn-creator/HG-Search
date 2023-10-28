import dgl 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, dropout, act=None, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        
        self.layer = homo_layer_dict[name](dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, homo_g, h):
        h = self.layer(homo_g, h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h

class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNConv, self).__init__()
        self.model = dgl.nn.pytorch.GraphConv(dim_in, dim_out, norm='both', bias=bias, allow_zero_in_degree=True)

    def forward(self, g, h):
        with g.local_scope():
            # g = dgl.add_reverse_edges(g)
            # g = dgl.remove_self_loop(g)
            # g = dgl.add_self_loop(g)
            h = self.model(g, h)
        return h


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = dgl.nn.pytorch.SAGEConv(dim_in, dim_out, aggregator_type='mean', bias=bias)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = dgl.nn.pytorch.GATConv(dim_in, dim_out, num_heads=kwargs['num_heads'], bias=bias, allow_zero_in_degree=True)

    def forward(self, g, h):
        # Note, falatten
        h = self.model(g, h).mean(1)
        return h


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        lin = nn.Sequential(nn.Linear(dim_in, dim_out, bias), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = dgl.nn.pytorch.GINConv(lin, 'max')

    def forward(self, g, h):
        h = self.model(g, h)
        return h
    

homo_layer_dict = {
    'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    'ginconv': GINConv,

}