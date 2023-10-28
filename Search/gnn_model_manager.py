from model_utils import TopAverage
import os 
import torch 
from model_utils import load_act 
import time 
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from openhgnn.trainerflow import build_flow

class CitationGNNManager(object):

    def __init__(self,args):

        self.args=args 
        self.reward_manager=TopAverage(10)
        # self.epochs=args.epochs 

    def read_config(self,actions):
        # specify the model family
        
        self.args.__setattr__('model',actions[0])
        self.args.__setattr__('model_name',actions[0])
        self.args.__setattr__('subgraph_extraction',actions[1])
        if self.args.model == 'homo_GNN':
            self.args.model_family = 'homogenization'
        elif self.args.model == 'general_HGNN':
            assert self.args.subgraph_extraction in ['relation', 'metapath', 'mixed']
            self.args.model_family = self.args.subgraph_extraction
        else:
            raise ValueError('Wrong model name or subgraph_extraction')
        
        fileNamePath = os.path.split(os.path.realpath(__file__))[0]
        if self.args.key == 'gnn_type':
            yamlPath = os.path.join(fileNamePath, 'config/{}/{}.yaml'.format(self.args.configfile, self.args.times))
        else:
            yamlPath=os.path.join(os.path.dirname(__file__),'../space4hgnn/config/test/has_bn/gcnconv_1.yaml')
        if self.args.gpu == -1:
            device = torch.device('cpu')
        elif self.args.gpu >= 0:
            if torch.cuda.is_available():
                device = torch.device('cuda', int(self.args.gpu))
            else:
                print("cuda is not available, please set 'gpu' -1")
        
        self.args.__setattr__('activation',actions[2])
        self.args.__setattr__('dropout',actions[3])
        self.args.__setattr__('feat',actions[4])
        self.args.__setattr__('gnn_type',actions[5])
        self.args.__setattr__('has_bn',actions[6])
        self.args.__setattr__('has_l2norm',actions[7])
        self.args.__setattr__('hidden_dim',actions[8])
        self.args.__setattr__('layers_gnn',actions[9])
        self.args.__setattr__('layers_post_mp',actions[10])
        self.args.__setattr__('layers_pre_mp',actions[11])
        self.args.__setattr__('loss_fn',actions[12])
        self.args.__setattr__('lr',actions[13])
        self.args.__setattr__('macro_func',actions[14])
        self.args.__setattr__('max_epoch',actions[15])
        self.args.__setattr__('num_heads',actions[16])
        self.args.__setattr__('optimizer',actions[17])
        self.args.__setattr__('stage_type',actions[18])

        # DBLP
        if self.args.dataset=="HGBn-DBLP":
            self.args.__setattr__('author_paper',actions[19])
            self.args.__setattr__('paper_author',actions[20])
            self.args.__setattr__('paper_term',actions[21])
            self.args.__setattr__('term_paper',actions[22])
            self.args.__setattr__('paper_venue',actions[23])
            self.args.__setattr__('venue_paper',actions[24])
        
        # IMDB
        if self.args.dataset=="HGBn-IMDB":
            self.args.__setattr__('actor_movie',actions[19])
            self.args.__setattr__('movie_actor',actions[20])
            self.args.__setattr__('director_movie',actions[21])
            self.args.__setattr__('movie_director',actions[22])
            self.args.__setattr__('keyword_movie',actions[23])
            self.args.__setattr__('movie_keyword',actions[24])
        
        # Freebase3
        if self.args.dataset=="HGBn-Freebase3":
            self.args.__setattr__('movie_actor',actions[19])
            self.args.__setattr__('actor_movie',actions[20])
            self.args.__setattr__('movie_direct',actions[21])
            self.args.__setattr__('direct_movie',actions[22])
            self.args.__setattr__('movie_writer',actions[23])
            self.args.__setattr__('writer_movie',actions[24])
        

        # AMiner
        if self.args.dataset=="HGBn-AMiner":
            self.args.__setattr__('paper_author',actions[19])
            self.args.__setattr__('author_paper',actions[20])
            self.args.__setattr__('paper_reference',actions[21])
            self.args.__setattr__('reference_paper',actions[22])

        
        if self.args.key in ['has_bn', 'has_l2norm']:
            self.args.value = self.args.value == "True"
        elif self.args.key in ['stage_type', 'activation', 'macro_func', 'gnn_type', 'optimizer']:
            self.args.value = self.args.value
        else:
            self.args.value = float(self.args.value)
            if self.args.value % 1 == 0:
                self.args.value = int(self.args.value)

        self.args.__setattr__(self.args.key, self.args.value)
        self.args.__setattr__('device', device)
        self.args.__setattr__('metric', "f1")

        path=os.path.join(os.path.dirname(__file__),'model_pt')
        if not os.path.exists(path):
            os.makedirs(path)
        self.args.__setattr__('_checkpoint', path)
        self.args.__setattr__('HGB_results_path', None)
        self.args.activation = load_act(self.args.activation)

    def evaluate(self,actions=None,format="two"):
        print("train action:",actions)
        t=time.localtime()
        self.args.seed=t
        self.read_config(actions)
        path = './space4hgnn/prediction/txt/{}/{}_{}/{}_{}_{}'.format(self.args.predictfile, self.args.key, self.args.value, self.args.model_family, self.args.gnn_type, self.args.times)
        if not os.path.exists(path):
            os.makedirs(path)
        self.args.HGB_results_path = '{}/{}_{}.txt'.format(path, self.args.dataset[5:],time.strftime('%Y-%m-%d %H:%M:%S',t))
        flow=build_flow(self.args,self.args.task)
        val_acc,test_acc=flow.run_model()
    
        return val_acc,test_acc
    
    def train(self,actions=None,format="two"):
        
        print("train action:",actions)
        t=time.localtime()
        self.args.seed=t
        self.read_config(actions)
        path = './space4hgnn/prediction/txt/{}/{}_{}/{}_{}_{}'.format(self.args.predictfile, self.args.key, self.args.value, self.args.model_family, self.args.gnn_type, self.args.times)
        if not os.path.exists(path):
            os.makedirs(path)
        self.args.HGB_results_path = '{}/{}_{}.txt'.format(path, self.args.dataset[5:],time.strftime('%Y-%m-%d %H:%M:%S',t))
        flow=build_flow(self.args,self.args.task)
        val_acc,test_acc=flow.run_model()
        reward=self.reward_manager.get_reward(val_acc['Macro_f1'])
        self.record_action_info(actions,reward,val_acc['Macro_f1'],test_acc['Macro_f1'])
        torch.cuda.empty_cache()
        return reward,val_acc['Macro_f1']
    

    def test_with_param(self,actions=None,format="two",with_retrain=False):
        return self.train(actions,format)

    def record_action_info(self,origin_action,reward,val_acc,test_acc):
        path=os.path.join(os.path.dirname(__file__),'../history_result/')
        with open(path+self.args.dataset+"_"+self.args.search_mode+self.args.submanager_log_file,"a") as file:
            file.write(str(origin_action))
            file.write(";")
            file.write(str(reward))
            file.write(";")
            file.write(str(val_acc))
            file.write(";")
            file.write(str(test_acc))
            file.write("\n")