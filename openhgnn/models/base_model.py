from abc import ABCMeta
import torch.nn as nn 

class BaseModel(nn.Module,metaclass=ABCMeta):
    
    @classmethod
    def build_model_from_args(cls,args,hg):

        raise NotImplementedError("Models must implement the build_model_from_args method")
    
    def __init__(self):
        super(BaseModel,self).__init__()

    def forward(self,*args):

        raise NotImplementedError 