from . import BaseFlow, register_flow
import dgl
import torch
import random 
import numpy as np 
from tqdm import tqdm
from ..models import build_model
from ..utils import EarlyStopping 

@register_flow("node_classification")
class NodeClassification(BaseFlow):

    def __init__(self,args):
        super(NodeClassification,self).__init__(args)
        self.args.category=self.task.dataset.category
        self.category=self.args.category
        self.num_classes=self.task.dataset.num_classes

        if not hasattr(self.task.dataset,'out_dim') or args.out_dim!=self.num_classes:
            args.out_dim = self.num_classes
        self.args.out_node_type=[self.category]
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()
        self.labels = self.task.get_labels().to(self.device)
        

    def preprocess(self):

        super(NodeClassification, self).preprocess()
    
    def run_model(self):
        self.preprocess()
        Stopper=EarlyStopping(self.args.patience,self._checkpoint)
        epoch_iter=tqdm(range(self.max_epoch))
        model_val_acc={'Macro_f1':0,'Micro_f1':0}
        best_performance={'Macro_f1':0,'Micro_f1':0}
        min_val_loss=float("inf")
        for epoch in epoch_iter:
            torch.cuda.empty_cache()
            train_loss=self._full_train_step()
            metric_dict,losses=self._full_test_step(modes=['train','valid'])
            val_acc=metric_dict['valid']
            val_loss=losses['valid']
            train_loss=losses['train']
            if val_loss<min_val_loss:
                min_val_loss=val_loss
                model_val_acc=val_acc
                metric_dict,losses=self._full_test_step(modes=['test'])
                best_performance=metric_dict['test']
            print(f"Epoch:{epoch},Train loss:{train_loss:.4f},Valid loss:{val_loss:.4f}",metric_dict)
            early_stop=Stopper.loss_step(val_loss,self.model)
            if early_stop:
                print('Early Stop!\t Epoch:'+str(epoch))
                break 
        Stopper.load_model(self.model)
        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        return model_val_acc,best_performance


    
    def _full_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        logits = self.model(self.hg, h_dict)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def _full_test_step(self, modes, logits=None):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            logits = logits if logits else self.model(self.hg, h_dict)[self.category]
            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idx
                elif mode == "valid":
                    masks[mode] = self.valid_idx
                elif mode == "test":
                    masks[mode] = self.test_idx
                    
            metric_dict = {key: self.task.evaluate(logits, mode=key) for key in masks}
            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metric_dict, loss_dict
