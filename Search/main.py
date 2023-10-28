import os 
import sys 
import argparse 
import torch 
import time 
from model_utils import set_random_seed 
import trainer 


def build_args():
    parser=argparse.ArgumentParser(description='hgnn_search')
    register_default_args(parser)
    args=parser.parse_args() 
    return args

def register_default_args(parser):
    parser.add_argument('--random_seed',type=int,default=123)
    parser.add_argument('--cuda',type=bool,default=True,required=False,
    help="run in cuda mode")
    parser.add_argument('--softmax_temperature',type=float,default=5.0)
    parser.add_argument('--tanh_c',type=float,default=2.5)
    parser.add_argument('--controller_optim',type=str,default='adam')
    parser.add_argument('--controller_lr',type=float,default=3.5e-4,
    help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--max_epoch',type=int,default=10)
    parser.add_argument('--controller_max_step',type=int,default=100,
                        help='step for controller parameters')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--format',type=str,default='two')
    # parser.add_argument('--dataset',type=str,default='ACM',required=False,
                        # help="The input dataset.")
    parser.add_argument('--submanager_log_file',type=str,default=f"sub_manager_logger_file_{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())}.txt")
    parser.add_argument('--time',type=str,default=f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())}")
    parser.add_argument('--entropy_coeff',type=float,default=1e-4)
    parser.add_argument('--ema_baseline_decay',type=float,default=0.95)
    # parser.add_argument("--epochs",type=int,default=300,help="number of training epochs")
    parser.add_argument('--max_save_num', type=int, default=5)
    parser.add_argument('--derive_num_sample', type=int, default=100)
    parser.add_argument('--derive_from_history', type=bool, default=True)
    parser.add_argument('--evaluation_metric',default='f1',type=str,help="the evaluation method")
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--derive_finally', type=bool, default=True)

   
    parser.add_argument('--model','-m',default='homo_GNN',type=str,help='name of models') 
    parser.add_argument('--model_name',default='homo_GNN',type=str,help='name of models')
    # parser.add_argument('--model','-m',default='homo_GNN',type=str,help='name of models') 
    # parser.add_argument('--model_name',default='homo_GNN',type=str,help='name of models')
    parser.add_argument('--subgraph_extraction', '-u', default='mixed', type=str, help='subgraph_extraction of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    parser.add_argument('--dataset', '-d', default='HGBn-DBLP', type=str, help='name of datasets')
    parser.add_argument('--dataset_name',default='HGBn-DBLP',type=str,help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--repeat', '-r', default='5', type=int, help='-1 means cpu')
    parser.add_argument('--times', '-s', default=1, type=int, help='which yaml file')
    parser.add_argument('--key', '-k', default='has_bn', type=str, help='attribute')
    parser.add_argument('--value', '-v', default='True', type=str, help='value')
    parser.add_argument('--configfile', '-c', default='test', type=str, help='The file path to load the configuration.')
    parser.add_argument('--predictfile', '-p', default='HGBn-DBLP', type=str, help='The file path to store predict files.')
    parser.add_argument('--mini_batch_flag',default=False,type=bool)
    parser.add_argument('--patience',default=20,type=int)
    parser.add_argument('--weight_decay',default=0.0001,type=float)


def main(args):
    args.logger=None 
    if args.cuda and not torch.cuda.is_available():
        args.cuda=False
    set_random_seed(args.random_seed)
    trnr=trainer.Trainer(args)
    trnr.train()

if __name__=="__main__":
    args=build_args()
    main(args)