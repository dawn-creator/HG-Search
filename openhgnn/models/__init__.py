from torch import nn 
import importlib
from .base_model import BaseModel

MODEL_REGISTRY={}
def register_model(name):

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate models ({})".format(name))
        if not issubclass(cls,BaseModel):
            raise ValueError("Model ({}: {}) must extend BaseModel".format(name, cls.__name__))
        MODEL_REGISTRY[name]=cls
        cls.model_name=name
        return cls 
    
    return register_model_cls 
from .homo_GNN import homo_GNN
from .general_HGNN import general_HGNN

SUPPORTED_MODELS={
   'homo_GNN': 'openhgnn.models.homo_GNN', 
   'general_HGNN': 'openhgnn.models.general_HGNN'
}



def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORTED_MODELS:
            importlib.import_module(SUPPORTED_MODELS[model])
        else:
            print(f"Failed to import {model} models.")
            return False
    return True 

def build_model(model):
    if isinstance(model,nn.Module):
        if not hasattr(model,'build_model_from_args'):
            def build_model_from_args(args,hg):
                return model 
            model.build_model_from_args=build_model_from_args
        return model 
    if not try_import_model(model):
        exit(1)
    return MODEL_REGISTRY[model]