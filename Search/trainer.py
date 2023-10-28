from graphnas_controller import SimpleNASController
from gnn_model_manager import CitationGNNManager
from model_utils import set_random_seed
import torch 
import numpy as np 
import tensor_utils as utils 
import os 
import random 
import dgl 


history=[]

def scale(value,last_k=10,scale_value=1):

    max_reward=np.max(np.abs(history[-last_k:]))
    if max_reward==0:
        return value 
    return scale_value/max_reward*value 

def _get_optimizer(name):
    if name.lower()=='sgd':
        optim=torch.optim.SGD
    elif name.lower()=='adam':
        optim=torch.optim.Adam
    return optim


class Trainer(object):
    
    def __init__(self,args):
        self.args=args
        self.cuda=args.cuda
        self.with_retrain=False
        self.build_model()
        controller_optimizer=_get_optimizer(self.args.controller_optim)
        self.controller_optim=controller_optimizer(self.controller.parameters(),lr=self.args.controller_lr)
        self.epoch=0 
        self.start_epoch=0 
        self.controller_step=0
        
    def build_model(self):
        from search_space import MacroSearchSpace
        search_space_cls=MacroSearchSpace()
        self.search_space=search_space_cls.get_search_space()
        if self.args.dataset=="HGBn-DBLP":
            self.search_space["author-paper"]=[1]
            self.search_space["paper-author"]=[1]
            self.search_space["paper-term"]=[1]
            self.search_space["term-paper"]=[1]
            self.search_space["paper-venue"]=[1]
            self.search_space["venue-paper"]=[1]
        if self.args.dataset=="HGBn-IMDB":
            self.search_space["actor->movie"]=[1]
            self.search_space["movie->actor"]=[1]
            self.search_space["director->movie"]=[1]
            self.search_space["movie->director"]=[1]
            self.search_space["keyword->movie"]=[1]
            self.search_space["movie->keyword"]=[1]
        if self.args.dataset=="HGBn-Freebase3":
            self.search_space["movie->actor"]=[1]
            self.search_space["actor->movie"]=[1]
            self.search_space["movie->direct"]=[1]
            self.search_space["dirctor->movie"]=[1]
            self.search_space["movie->writer"]=[1]
            self.search_space["writer->movie"]=[1]    
        if self.args.dataset=="HGBn-AMiner":
            self.search_space["paper->author"]=[1]
            self.search_space["author->paper"]=[1]
            self.search_space["paper->reference"]=[1]
            self.search_space["reference->paper"]=[1]

        self.action_list=search_space_cls.generate_action_list()
        # build RNN controller 
        # generate HGNN sequence
        self.controller=SimpleNASController(self.args,action_list=self.action_list,
                                            search_space=self.search_space,
                                            cuda=self.args.cuda)
        ## HGNN Framework
        self.submodel_manager=CitationGNNManager(self.args)
        if self.cuda:
            self.controller.cuda()
    
    def train(self):
        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.train_controller()
        
        if self.args.derive_finally:
            best_actions=self.derive_from_history()
            print("best structure:"+str(best_actions))
        # self.train_ablation_study()

    def derive_from_history(self):

        path=os.path.join(os.path.dirname(__file__),'../history_result/')
        with open(path+self.args.dataset+"_"+self.args.search_mode+self.args.submanager_log_file,"r") as f:
             lines=f.readlines()
        
        results=[]
        best_val_score="0"
        for line in lines:
            actions=line[:line.index(";")]
            val_score=line.split(";")[-2]
            results.append((actions,val_score))
        results.sort(key=lambda x:x[-1],reverse=True)
        best_structure=""
        best_score=0 
        for actions in results[:100]:
            actions=eval(actions[0])
            val_scores_list=[]
            val_scores_list_micro=[]
            for i in range(10):
                random.seed(i)
                np.random.seed(i)
                torch.manual_seed(i)
                torch.cuda.manual_seed(i)
                dgl.seed(i)
                val_acc,test_acc=self.submodel_manager.evaluate(actions)
                val_scores_list.append(val_acc['Macro_f1'])
                

            temp_score=np.mean(val_scores_list)
            temp_score_std=np.std(val_scores_list)
            if temp_score>best_score:
                best_score=temp_score
                best_structure=actions



        # best_structure=['general_HGNN', 'relation', 'tanh', 0.6, 0, 'gatconv', True, True, 128, 5, 3, 1, 'dot-product', 0.001, 'sum', 100, 4, 'Adam', 'skipsum', 0, 1, 1, 1, 1, 1]
        # train from scratch to get the final score
        test_scores_list_macro = []
        test_scores_list_micro=[]
        for i in range(100):
            set_random_seed(i)
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list_macro.append(test_acc['Macro_f1'])
            test_scores_list_micro.append(test_acc['Micro_f1'])
            print(i)
        print(f"best results: {best_structure}  Macro-f1: {np.mean(test_scores_list_macro):.8f} +/- {np.std(test_scores_list_macro)}  Micro-f1: {np.mean(test_scores_list_micro):.8f} +/- {np.std(test_scores_list_micro)}")
        return best_structure
    
    def evaluate_sample(self):

        if self.args.dataset=="HGBn-DBLP":
            best_structure=['general_HGNN', 'relation', 'lrelu', 0.5, 0, 'gatconv', False, True, 64, 4, 3, 1, 'distmult', 0.001, 'attention', 400, 1, 'Adam', 'stack', 1, 1, 1, 1, 1, 1]
        if self.args.dataset=="HGBn-IMDB":
            best_structure=['general_HGNN', 'relation', 'tanh', 0.6, 0, 'gatconv', True, True, 128, 2, 2, 1, 'distmult', 0.001, 'sum', 100, 2, 'Adam', 'stack', 1, 1, 1, 1, 1, 1] 
        if self.args.dataset=="HGBn-Freebase3":
            best_structure=['general_HGNN', 'mixed', 'tanh', 0.4, 2, 'gcnconv', False, True, 128, 2, 1, 1, 'distmult', 0.001, 'sum', 100, 2, 'Adam', 'skipconcat', 1, 1, 1, 1, 1, 1]  
        if self.args.dataset=="HGBn-AMiner":
            best_structure=['general_HGNN', 'mixed', 'elu', 0.6, 2, 'gcnconv', False, False, 128, 4, 1, 2, 'distmult', 0.001, 'max', 100, 8, 'Adam', 'stack', 1, 1, 1, 1] 
        test_scores_list_macro = []
        test_scores_list_micro=[]
        for i in range(100):
            random.seed(i)
            np.random.seed(i)
            torch.manual_seed(i)
            torch.cuda.manual_seed(i)
            dgl.seed(i)
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list_macro.append(test_acc['Macro_f1'])
            test_scores_list_micro.append(test_acc['Micro_f1'])
            print(i)

        print(f"best results: {best_structure}  Macro-f1: {np.mean(test_scores_list_macro):.8f} +/- {np.std(test_scores_list_macro)}  Micro-f1: {np.mean(test_scores_list_micro):.8f} +/- {np.std(test_scores_list_micro)}")
        return best_structure
    


    def train_ablation_study(self):

        def record_action_info(origin_action,val_acc,test_acc):
            with open("/home/weidu/shm/OpenHGNN-main_nc/space4hgnn/Experiment_Data_valid/HGBn-DBLP_macrosub_manager_logger_file_2023-07-29_02_33_17.txt","a") as file:
                file.write(str(origin_action))
                file.write(";")
                file.write(str(val_acc))
                file.write(";")
                file.write(str(test_acc))
                file.write("\n")

        best_structure_ACM=['homo_GNN', 'relation', 'lrelu', 0.5, 0, 'sageconv', True, True, 128, 6, 1, 1, 'dot-product', 0.001, 'attention', 100, 8, 'Adam', 'stack', 1, 1, 1, 1, 1, 1, 1, 0] 
        best_score_macro=0
        best_score_micro=0
        best_score_macro_std=0
        best_score_micro_std=0
        with open("/home/weidu/shm/OpenHGNN-main_nc/space4hgnn/Experiment_data/HGBn-DBLP_macrosub_manager_logger_file_2023-07-29_02_33_17.txt","r") as f:
                lines=f.readlines()
        for line in lines:
            action=eval(line[:line.index(";")])
            best_structure_ACM[0]=action[0]
            best_structure_ACM[1]=action[1]
            best_structure_ACM[2]=action[2]
            best_structure_ACM[3]=action[3]
            best_structure_ACM[4]=action[4]
            best_structure_ACM[5]=action[5]
            best_structure_ACM[6]=action[6]
            best_structure_ACM[7]=action[7]
            best_structure_ACM[8]=action[8]
            best_structure_ACM[9]=action[9]
            best_structure_ACM[10]=action[10]
            best_structure_ACM[11]=action[11]
            best_structure_ACM[12]=action[12]
            best_structure_ACM[13]=action[13]
            best_structure_ACM[14]=action[14]
            best_structure_ACM[15]=action[15]
            best_structure_ACM[16]=action[16]
            best_structure_ACM[17]=action[17]
            best_structure_ACM[18]=action[18]
            best_structure_ACM[19]=action[19]
            best_structure_ACM[20]=action[20]
            best_structure_ACM[21]=action[21]
            best_structure_ACM[22]=action[22]
            best_structure_ACM[23]=action[23]
            best_structure_ACM[24]=action[24]
            # best_structure_ACM[25]=action[25]
            # best_structure_ACM[26]=action[26]
            val_acc,test_acc=self.submodel_manager.evaluate(best_structure_ACM)
            record_action_info(best_structure_ACM,val_acc['Macro_f1'],val_acc['Micro_f1'])
    
    def train_controller(self):
        """
           Train controller to find better structure
        """
        print("*"*35,"training controller","*"*35)
        model=self.controller 
        model.train()
        baseline=None 
        adv_history=[]
        total_loss=0
        
        for step in range(self.args.controller_max_step):
            # sample graphnas
            structure_list,log_probs,entropies=self.controller.sample(with_details=True)

            # calculate reward
            np_entropies=entropies.data.cpu().numpy()
            results=self.get_reward(structure_list,np_entropies)
            torch.cuda.empty_cache()
            rewards=results 
            if baseline is None:
               baseline=rewards
            else:
               decay=self.args.ema_baseline_decay
               baseline=decay*baseline+(1-decay)*rewards
            
            adv=rewards-baseline
            history.append(adv)
            adv=scale(adv,scale_value=0.5)
            adv_history.extend(adv)
            adv=utils.get_variable(adv,self.cuda,requires_grad=False)
            # policy loss
            loss=-log_probs*adv 
            loss=loss.sum() # or loss.mean()
            
            # update 
            self.controller_optim.zero_grad()
            loss.backward()
            self.controller_optim.step()

            total_loss+=utils.to_item(loss.data)

            self.controller_step+=1
            torch.cuda.empty_cache()

        print("*"*35,"training controller over","*"*35)  


    def get_reward(self,gnn_list,entropies):
        
        reward_list=[]
        for gnn in gnn_list:
            reward=self.submodel_manager.test_with_param(gnn,format=self.args.format,
                                                         with_retrain=self.with_retrain)
            if reward is None:
               reward=0
            else:
               reward=reward[1]

            reward_list.append(reward)
        
        rewards=reward_list+self.args.entropy_coeff*entropies

        return rewards