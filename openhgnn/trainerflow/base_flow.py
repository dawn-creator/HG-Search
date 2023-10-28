import os 
from abc import ABC,abstractclassmethod 
import numpy as np 
import time 
import torch 
from ..tasks import build_task 
from ..utils import get_nodes_dict
from ..layers.HeteroLinear import HeteroFeature

class BaseFlow(ABC):
    candidate_optimizer={
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'Adadelta': torch.optim.Adadelta
    }

    def __init__(self,args):

        super(BaseFlow,self).__init__()


        self.evaluator = None
        self.evaluate_interval = 1
        if hasattr(args, '_checkpoint'):
            self._checkpoint = os.path.join(args._checkpoint, f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())}_{args.model_name}_{args.dataset_name}.pt")
        else:
            if hasattr(args, 'load_from_pretrained'):
                self._checkpoint = os.path.join(args.output_dir,f"{args.model_name}_{args.dataset_name}_{args.task}.pt")
            else:
                self._checkpoint = None
        if not hasattr(args, 'HGB_results_path') and args.dataset_name[:3] == 'HGB':
            args.HGB_results_path = os.path.join(args.output_dir,"{}_{}_{}.txt".format(args.model_name, args.dataset_name[5:],args.seed))

        self.args = args
        self.logger = self.args.logger
        self.model_name = args.model_name
        self.model = args.model
        self.device = args.device
        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)
        self.args.meta_paths_dict = self.task.dataset.meta_paths_dict
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.optimizer = None
        self.loss_fn = self.task.get_loss_fn()

    def preprocess(self):

        if hasattr(self.args, 'activation'):
            if hasattr(self.args.activation, 'weight'):
                import torch.nn as nn
                act = nn.PReLU()
            else:
                act = self.args.activation
        else:
            act = None

        self.feature_preprocess(act)
        self.optimizer.add_param_group({'params': self.input_feature.parameters()})
        self.model.add_module('input_feature', self.input_feature)

    def feature_preprocess(self, act):
        if self.hg.ndata.get('h', {}) == {} or self.args.feat == 2:
            if self.hg.ndata.get('h', {}) == {}:
                print('Assign embedding as features, because hg.ndata is empty.')
            else:
                print('feat2, drop features!')
                self.hg.ndata.pop('h')
            self.input_feature = HeteroFeature({}, get_nodes_dict(self.hg), self.args.hidden_dim,
                                               act=act).to(self.device)
        elif self.args.feat == 0:
            self.input_feature = self.init_feature(act)
        elif self.args.feat == 1:
            if self.args.task != 'node_classification':
                print('\'feat 1\' is only for node classification task, set feat 0!')
                self.input_feature = self.init_feature(act)
            else:
                h_dict = self.hg.ndata.pop('h')
                print('feat1, preserve target nodes!')
                self.input_feature = HeteroFeature({self.category: h_dict[self.category]}, get_nodes_dict(self.hg), self.args.hidden_dim,
                                                   act=act).to(self.device)
    
    def init_feature(self, act):
        print("Feat is 0, nothing to do!")
        if isinstance(self.hg.ndata['h'], dict):
           
            input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg),
                                               self.args.hidden_dim, act=act,need_distort=False).to(self.device)
            
        elif isinstance(self.hg.ndata['h'], torch.Tensor):
            # The heterogeneous only contains one node type.
            input_feature = HeteroFeature({self.hg.ntypes[0]: self.hg.ndata['h']}, get_nodes_dict(self.hg),
                                               self.args.hidden_dim, act=act,need_distort=False).to(self.device)
        return input_feature