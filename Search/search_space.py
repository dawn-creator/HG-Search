import torch 

class MacroSearchSpace(object):
    def __init__(self,search_space=None):
        if search_space:
            self.search_space=search_space
        else:
            self.search_space={
               "model":["general_HGNN","homo_GNN"],
                "subgraph_extraction":["relation","mixed"],
                "activation":["relu","lrelu","elu","tanh","sigmoid"],
                "dropout":[0.3,0.4,0.5,0.6],
                "feat":[0,1,2],
                "gnn_type":["gcnconv","gatconv","ginconv","sageconv"],
                "has_bn":[True,False],
                "has_l2norm":[True,False],
                "hidden_dim":[8,16,32,64,128],
                "layers_gnn":[1,2,3,4,5,6],
                "layers_post_mp":[1,2,3],
                "layers_pre_mp":[1],
                "loss_fn":["distmult"],
                "lr":[0.01,0.001],
                "macro_func":["attention","sum","mean","max"],
                "max_epoch":[100,200,400],
                "num_heads":[1,2,4,8],
                "optimizer":["Adam"],
                "stage_type":["stack","skipsum","skipconcat"],   
            }


    def get_search_space(self):
        return self.search_space
    
    def generate_action_list(self):
        action_list=list(self.search_space.keys())
        return action_list 