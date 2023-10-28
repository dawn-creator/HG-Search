import numpy as np 
import torch.nn as nn 
import dgl 
import torch 
import random 

act_dict = {
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'sigmoid': nn.Sigmoid(),
    'lrelu': nn.LeakyReLU(negative_slope=0.5),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'prelu': nn.PReLU(),
    'selu': nn.SELU(),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5),
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.seed(seed)

def load_act(act):
    act = act_dict.get(act, None)
    if act is None:
        raise ValueError('No corresponding activation')
    return act

class FixedList(list):
    def __init__(self,size=10):
        super(FixedList,self).__init__()
        self.size=size 
    
    def append(self,obj):
        if len(self)>=self.size:
            self.pop(0)
        super().append(obj)

class TopAverage(object):
    def __init__(self,top_k=10):
        self.scores=[]
        self.top_k=top_k
    
    def get_top_average(self):
        if len(self.scores)>0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self,score):
        if len(self.scores)>0:
            avg=np.mean(self.scores)
        else:
            avg=0
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores=self.scores[:self.top_k]
        return avg
    
    def get_reward(self,score):
        reward=score-self.get_average(score)
        return np.clip(reward,-0.5,0.5)