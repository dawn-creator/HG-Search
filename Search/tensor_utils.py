import numpy as np
import torch
from torch.autograd import Variable 
import logging 
import os 



def get_variable(inputs,cuda=False,**kwargs):
    if type(inputs) in [list,np.ndarray]:
        inputs=torch.Tensor(inputs)
    if cuda:
        out=Variable(inputs.cuda(),**kwargs)
    else:
        out=Variable(inputs,**kwargs)
    return out

def to_item(x):
    if isinstance(x,(float,int)):
        return x
    
    if float(torch.__version__[0:3])<0.4:
        assert (x.dim()==1) and (len(x)==1)
        return x[0]

    return x.item()