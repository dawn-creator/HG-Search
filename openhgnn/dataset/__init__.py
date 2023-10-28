from .base_dataset import BaseDataset 
from .hgb_dataset import HGBDataset 
DATASET_REGISTRY={}
hgbn_datasets=['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB','HGBn-Freebase3','HGBn-AMiner']

def register_dataset(name):

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls,BaseDataset):
            raise ValueError("Dataset ({}:{}) must extend cogdl.data.Dataset".format(name,cls.__name__))
        DATASET_REGISTRY[name]=cls
        return cls 
    
    return register_dataset_cls 

from .NodeClassificationDataset import NodeClassificationDataset 





def build_dataset(dataset,task,*args,**kwargs):
    
    if dataset in hgbn_datasets:
        _dataset='HGBn_node_classification'
    return DATASET_REGISTRY[_dataset](dataset,logger=kwargs['logger'])