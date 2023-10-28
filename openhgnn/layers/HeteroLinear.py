import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class GeneralLinear(nn.Module):

    def __init__(self, in_features, out_features, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(GeneralLinear, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = nn.Linear(in_features, out_features, bias=not has_bn)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(out_features))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch_h: torch.Tensor) -> torch.Tensor:
        
        batch_h = self.layer(batch_h)
        batch_h = self.post_layer(batch_h)
        if self.has_l2norm:
            batch_h = F.normalize(batch_h, p=2, dim=1)
        return batch_h


class HeteroLinearLayer(nn.Module):

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(HeteroLinearLayer, self).__init__()

        self.layer = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            self.layer[name] = GeneralLinear(in_features=linear_dim[0], out_features=linear_dim[1], act=act,
                                                  dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
    
    def forward(self, dict_h: dict) -> dict:
        
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layer[name](batch_h)
        return new_h
    

class HeteroFeature(nn.Module):

    def __init__(self, h_dict, n_nodes_dict, embed_size, act=None, need_trans=True, all_feats=True,need_distort=False):
        super(HeteroFeature, self).__init__()
        self.n_nodes_dict = n_nodes_dict
        self.embed_size = embed_size
        self.h_dict = h_dict
        self.need_trans = need_trans
        self.embed_dict = nn.ParameterDict()
        linear_dict = {}
        for ntype, n_nodes in self.n_nodes_dict.items():
            h = h_dict.get(ntype)
            if h is None:
                if all_feats:
                    embed = nn.Parameter(torch.FloatTensor(n_nodes, self.embed_size))
                    nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
                    self.embed_dict[ntype] = embed
            else:
                linear_dict[ntype] = [h.shape[1], self.embed_size]
        if need_distort:
            h_dict=self.data_distort(h_dict)
        if need_trans:
            self.hetero_linear = HeteroLinearLayer(linear_dict, act=act)
    

    def data_distort(self,h_dict):
        for ntype, _ in self.n_nodes_dict.items():
            if h_dict.get(ntype) is None:
                self.drop_feature(self.embed_dict[ntype].data) 
            else:
                self.drop_feature(h_dict[ntype])
        return h_dict
    
    def drop_feature(self,x,drop_prob=0.2):
        drop_mask=torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0,1)<drop_prob
        x[:,drop_mask]=0

    def forward(self):
        out_dict = {}
        for ntype, _ in self.n_nodes_dict.items():
            if self.h_dict.get(ntype) is None:
                out_dict[ntype] = self.embed_dict[ntype]
        if self.need_trans:
            out_dict.update(self.hetero_linear(self.h_dict))
        else:
            out_dict.update(self.h_dict)
        return out_dict

    def forward_nodes(self, nodes_dict):
        out_feature = {}
        for ntype, nid in nodes_dict.items():
            if self.h_dict.get(ntype) is None:
                out_feature[ntype] = self.embed_dict[ntype][nid]
            else:
                if self.need_trans:
                    out_feature[ntype] = self.hetero_linear(self.h_dict)[ntype][nid]
                else:
                    out_feature[ntype] = self.h_dict[ntype][nid]
        return out_feature

class HeteroMLPLayer(nn.Module):

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, final_act=False, **kwargs):
        super(HeteroMLPLayer, self).__init__()
        self.layers = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            nn_list = []
            n_layer = len(linear_dim) - 1
            for i in range(n_layer):
                in_dim = linear_dim[i]
                out_dim = linear_dim[i+1]
                if i == n_layer - 1:
                    if final_act:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act,
                                              droPout=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                    else:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=None,
                                          droPout=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                else:
                    layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act,
                                      droPout=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

                nn_list.append(layer)
            self.layers[name] = nn.Sequential(*nn_list)

    def forward(self, dict_h):
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layers[name](batch_h)
        return new_h