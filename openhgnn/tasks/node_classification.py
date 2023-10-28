import torch.nn.functional as F 
import torch.nn as nn 
from . import BaseTask
from ..dataset import build_dataset 
from ..utils import Evaluator
from . import register_task 

@register_task("node_classification")
class NodeClassification(BaseTask):
    def __init__(self, args):
        super(NodeClassification, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, 'node_classification', logger=self.logger)
        self.logger=args.logger 
        if hasattr(args,'validation'):
            self.train_idx,self.val_idx,self.test_idx=self.dataset.get_split(args.validation)
        else:
            self.train_idx,self.val_idx,self.test_idx=self.dataset.get_split()
        self.evaluator=Evaluator(args.seed)
        self.labels=self.dataset.get_labels()
        self.multi_label=self.dataset.multi_label
        if hasattr(args, 'evaluation_metric'):
            self.evaluation_metric = args.evaluation_metric
        else:
            if args.dataset in ['aifb', 'mutag', 'bgs', 'am']:
                self.evaluation_metric = 'acc'
            else:
                self.evaluation_metric = 'f1'
    
    def get_graph(self):
        return self.dataset.g 
    
    def get_loss_fn(self):
        if self.multi_label:
            return nn.BCEWithLogitsLoss()
        return F.cross_entropy
    
    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx
    
    def get_labels(self):
        return self.labels
    
    def evaluate(self, logits, mode='test', info=True):
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx

        if self.multi_label:
            pred = (logits[mask].cpu().numpy() > 0).astype(int)
        else:
            pred = logits[mask].argmax(dim=1).to('cpu')
        
        if self.evaluation_metric=='f1':
            f1_dict = self.evaluator.f1_node_classification(self.labels[mask], pred)
            return f1_dict
        else:
            raise ValueError('The evaluation metric is not supported!')