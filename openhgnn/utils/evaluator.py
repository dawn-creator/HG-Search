from sklearn.metrics import f1_score

class Evaluator():
    def __init__(self,seed):
        self.seed=seed
    
    def f1_node_classification(self,y_label,y_pred):
        macro_f1=f1_score(y_label,y_pred,average='macro')
        micro_f1=f1_score(y_label,y_pred,average='micro')
        return dict(Macro_f1=macro_f1,Micro_f1=micro_f1)
    

    