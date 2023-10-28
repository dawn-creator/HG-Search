import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import copy
import torch as th

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

from data_loader import data_loader 


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,  # Learning rate
    'num_heads': [8],  # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
}

sampling_configure = {
    'batch_size': 20
}


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm(feat_type=0):
    dl=data_loader(os.path.join(os.path.dirname(__file__),'ACM'))
    # link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    link_type_dic={0:'paper-cite-paper',1:'paper-ref-paper',2:'paper-author',3:'author-paper',
                   4:'paper-subject',5:'subject-paper',6:'paper-term',7:'term-paper'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        if dl.links['meta'][link_type][0]==0:
            src_type='paper'
        elif dl.links['meta'][link_type][0]==1:
            src_type='author'
        elif dl.links['meta'][link_type][0]==2:
            src_type='subject'
        else:
            src_type='term'
        if dl.links['meta'][link_type][1]==0:
            dst_type='paper'
        elif dl.links['meta'][link_type][1]==1:
            dst_type='author'
        elif dl.links['meta'][link_type][1]==2:
            dst_type='subject'
        else:
            dst_type='term'
        # data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        '''
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        '''
        
        if link_type==0:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        elif link_type==1:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        elif link_type==2:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-3025)
        elif link_type==3:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-3025,dl.links['data'][link_type].nonzero()[1])
        elif link_type==4:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-8984)
        elif link_type==5:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-8984,dl.links['data'][link_type].nonzero()[1]) 
        elif link_type==6:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-9040)
        else:
           data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-9040,dl.links['data'][link_type].nonzero()[1])
        
    hg = dgl.heterograph(data_dic)

    hg.nodes['paper'].data['test_mask']=torch.tensor(dl.labels_test['mask'][:paper_num],dtype=torch.uint8)
    hg.nodes['paper'].data['train_mask']=torch.tensor(dl.labels_train['mask'][:paper_num],dtype=torch.uint8)
    hg.nodes['paper'].data['h']=torch.tensor(dl.nodes['attr'][0]).to(torch.float32)
    hg.nodes['author'].data['h']=torch.tensor(dl.nodes['attr'][1]).to(torch.float32)
    hg.nodes['subject'].data['h']=torch.tensor(dl.nodes['attr'][2]).to(torch.float32)


    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)
    labels=labels.to(torch.float32)

    hg.nodes['paper'].data['label']=labels
    num_class=3
    return hg,features,labels,num_class,train_indices,valid_indices,test_indices,train_mask,valid_mask,test_mask
    # return hg 

    metapaths_dict={
                    'PAP': [('paper', 'paper-author', 'author'), 
                            ('author', 'author-paper', 'paper')],
                    'PSP': [('paper', 'paper-subject', 'subject'),
                            ('subject', 'subject-paper', 'paper')],
                    'PcPAP': [('paper', 'paper-cite-paper', 'paper'),
                              ('paper', 'paper-author', 'author'),
                              ('author', 'author-paper', 'paper')],
                    'PcPSP': [('paper', 'paper-cite-paper', 'paper'),
                              ('paper', 'paper-subject', 'subject'),
                              ('subject', 'subject-paper', 'paper')],
                    'PrPAP': [('paper', 'paper-ref-paper', 'paper'),
                              ('paper', 'paper-author', 'author'),
                              ('author', 'author-paper', 'paper')],
                    'PrPSP': [('paper', 'paper-ref-paper', 'paper'),
                              ('paper', 'paper-subject', 'subject'),
                              ('subject', 'subject-paper', 'paper')]
                    }
    graph_data={}
    for key, mp in metapaths_dict.items():
        mp_g = dgl.metapath_reachable_graph(hg, mp)
        n_edge = mp_g.canonical_etypes[0]
        graph_data[(n_edge[0], key, n_edge[2])] = mp_g.edges()
    new_hg=dgl.heterograph(graph_data)

    # paper feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    # author labels
    
    

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)
    labels=labels.to(torch.float32)

    hg.nodes['paper'].data['label']=labels
    
    return hg 
    num_classes = 3

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    # meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    meta_paths=[['paper-author','author-paper'],['paper-subject','subject-paper'],['paper-cite-paper','paper-author','author-paper'],
                ['paper-cite-paper','paper-subject','subject-paper'],['paper-ref-paper','paper-author','author-paper'],
                ['paper-ref-paper','paper-subject','subject-paper ']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_freebase(feat_type=1):
    dl=data_loader(os.path.join(os.path.dirname(__file__),'Freebase'))

    link_type_dic={
                   0:'BOOK-and-BOOK',1:'BOOK-to-FILM',2:'BOOK-on-SPORTS',3:'BOOK-on-LOCATION',4:'BOOK-about-ORGANIZATION',
                   5:'FILM-and-FILM',6:'MUSIC-in-BOOK',7:'MUSIC-in-FILM',8:'MUSIC-and-MUSIC',
                   9:'MUSIC-for-SPORTS',10:'MUSIC-on-LOCATION',11:'SPORTS-in-FILM',12:'SPORTS-and-SPORTS',
                   13:'SPORTS-on-LOCATION',14:'PEOPLE-to-BOOK',15:'PEOPLE-to-FILM',16:'PEOPLE-to-MUSIC',
                   17:'PEOPLE-to-SPORTS',18:'PEOPLE-and-PEOPLE',19:'PEOPLE-on-LOCATION',20:'PEOPLE-in-ORGANIZATION',
                   21:'PEOPLE-in-BUSINESS',22:'LOCATION-in-FILM',23:'LOCATION-and-LOCATION',24:'ORGANIZATION-in-FILM',
                   25:'ORGANIZATION-to-MUSIC',26:'ORGANIZATION-to-SPORTS',27:'ORGANIZATION-on-LOCATION',28:'ORGANIZATION-and-ORGANIZATION',
                   29:'ORGANIZATION-for-BUSINESS',30:'BUSINESS-about-BOOK',31:'BUSINESS-about-FILM',32:'BUSINESS-about-MUSIC',
                   33:'BUSINESS-about-SPORTS',34:'BUSINESS-on-LOCATION',35:'BUSINESS-and-BUSINESS'}
              
    book_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        if link_type==0:
            src_type='BOOK'
            dst_type='BOOK'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = dl.links['data'][link_type].T.nonzero()
            '''
        elif link_type==1:
            src_type='BOOK'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1])
            '''
        elif link_type==2:
            src_type='BOOK'
            dst_type='SPORTS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-142180)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-142180,dl.links['data'][link_type].T.nonzero()[1])
            '''
        elif link_type==3:
            src_type='BOOK'
            dst_type='LOCATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-160846)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-160846,dl.links['data'][link_type].T.nonzero()[1])
            '''
        elif link_type==4:
            src_type='BOOK'
            dst_type='ORGANIZATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-170214)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-170214,dl.links['data'][link_type].T.nonzero()[1])
            '''
        elif link_type==5:
            src_type='FILM'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-40402,dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1]-40402)
            '''
        elif link_type==6:
            src_type='MUSIC'
            dst_type='BOOK'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-59829,dl.links['data'][link_type].nonzero()[1])
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0],dl.links['data'][link_type].T.nonzero()[1]-59829)
            '''
        elif link_type==7:
            src_type='MUSIC'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-59829,dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1]-59829)
            '''
        elif link_type==8:
            src_type='MUSIC'
            dst_type='MUSIC'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-59829,dl.links['data'][link_type].nonzero()[1]-59829)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-59829,dl.links['data'][link_type].T.nonzero()[1]-59829)
            '''
        elif link_type==9:
            src_type='MUSIC'
            dst_type='SPORTS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-59829,dl.links['data'][link_type].nonzero()[1]-142180)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-142180,dl.links['data'][link_type].T.nonzero()[1]-59829)
            '''
        elif link_type==10:
            src_type='MUSIC'
            dst_type='LOCATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-59829,dl.links['data'][link_type].nonzero()[1]-160846)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-160846,dl.links['data'][link_type].T.nonzero()[1]-59829)
            '''
        elif link_type==11:
            src_type='SPORTS'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-142180,dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1]-142180)
            '''
        elif link_type==12:
            src_type='SPORTS'
            dst_type='SPORTS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-142180,dl.links['data'][link_type].nonzero()[1]-142180)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-142180,dl.links['data'][link_type].T.nonzero()[1]-142180)
            '''
        elif link_type==13:
            src_type='SPORTS'
            dst_type='LOCATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-142180,dl.links['data'][link_type].nonzero()[1]-160846)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-160846,dl.links['data'][link_type].T.nonzero()[1]-142180)
            '''
        elif link_type==14:
            src_type='PEOPLE'
            dst_type='BOOK'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1])
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0],dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==15:
            src_type='PEOPLE'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==16:
            src_type='PEOPLE'
            dst_type='MUSIC'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1]-59829)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-59829,dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==17:
            src_type='PEOPLE'
            dst_type='SPORTS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1]-142180)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-142180,dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==18:
            src_type='PEOPLE'
            dst_type='PEOPLE'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1]-143205)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-143205,dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==19:
            src_type='PEOPLE'
            dst_type='LOCATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1]-160846)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-160846,dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==20:
            src_type='PEOPLE'
            dst_type='ORGANIZATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1]-170214)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-170214,dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==21:
            src_type='PEOPLE'
            dst_type='BUSINESS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-143205,dl.links['data'][link_type].nonzero()[1]-172945)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-172945,dl.links['data'][link_type].T.nonzero()[1]-143205)
            '''
        elif link_type==22:
            src_type='LOCATION'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-160846,dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1]-160846)
            '''
        elif link_type==23:
            src_type='LOCATION'
            dst_type='LOCATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-160846,dl.links['data'][link_type].nonzero()[1]-160846)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-160846,dl.links['data'][link_type].T.nonzero()[1]-160846)
            '''
        elif link_type==24:
            src_type='ORGANIZATION'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-170214,dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1]-170214)
            '''
        elif link_type==25:
            src_type='ORGANIZATION'
            dst_type='MUSIC'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-170214,dl.links['data'][link_type].nonzero()[1]-59829)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-59829,dl.links['data'][link_type].T.nonzero()[1]-170214)
            '''
        elif link_type==26:
            src_type='ORGANIZATION'
            dst_type='SPORTS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-170214,dl.links['data'][link_type].nonzero()[1]-142180)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-142180,dl.links['data'][link_type].T.nonzero()[1]-170214)
            '''
        elif link_type==27:
            src_type='ORGANIZATION'
            dst_type='LOCATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-170214,dl.links['data'][link_type].nonzero()[1]-160846)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-160846,dl.links['data'][link_type].T.nonzero()[1]-170214)
            '''
        elif link_type==28:
            src_type='ORGANIZATION'
            dst_type='ORGANIZATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-170214,dl.links['data'][link_type].nonzero()[1]-170214)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-170214,dl.links['data'][link_type].T.nonzero()[1]-170214)
            '''
        elif link_type==29:
            src_type='ORGANIZATION'
            dst_type='BUSINESS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-170214,dl.links['data'][link_type].nonzero()[1]-172945)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-172945,dl.links['data'][link_type].T.nonzero()[1]-170214)
            '''
        elif link_type==30:
            src_type='BUSINESS'
            dst_type='BOOK'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-172945,dl.links['data'][link_type].nonzero()[1])
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0],dl.links['data'][link_type].T.nonzero()[1]-172945)
            '''
        elif link_type==31:
            src_type='BUSINESS'
            dst_type='FILM'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-172945,dl.links['data'][link_type].nonzero()[1]-40402)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-40402,dl.links['data'][link_type].T.nonzero()[1]-172945)
            '''
        elif link_type==32:
            src_type='BUSINESS'
            dst_type='MUSIC'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-172945,dl.links['data'][link_type].nonzero()[1]-59829)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-59829,dl.links['data'][link_type].T.nonzero()[1]-172945)
            '''
        elif link_type==33:
            src_type='BUSINESS'
            dst_type='SPORTS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-172945,dl.links['data'][link_type].nonzero()[1]-142180)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-142180,dl.links['data'][link_type].T.nonzero()[1]-172945)
            '''
        elif link_type==34:
            src_type='BUSINESS'
            dst_type='LOCATION'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-172945,dl.links['data'][link_type].nonzero()[1]-160846)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-160846,dl.links['data'][link_type].T.nonzero()[1]-172945)
            '''
        else:
            src_type='BUSINESS'
            dst_type='BUSINESS'
            
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-172945,dl.links['data'][link_type].nonzero()[1]-172945)
            # reverse
            '''
            if link_type_dic[link_type + 36][0] != '-':
                data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = (dl.links['data'][link_type].T.nonzero()[0]-172945,dl.links['data'][link_type].T.nonzero()[1]-172945)
            '''
    hg = dgl.heterograph(data_dic)
    
    hg.nodes['BOOK'].data['test_mask']=torch.tensor(dl.labels_test['mask'][:book_num],dtype=torch.uint8)
    hg.nodes['BOOK'].data['train_mask']=torch.tensor(dl.labels_train['mask'][:book_num],dtype=torch.uint8)
    labels = dl.labels_test['data'][:book_num] + dl.labels_train['data'][:book_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)
    # labels=labels.to(torch.float32)
    hg.nodes['BOOK'].data['label']=labels
    return hg 
    
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        indices = np.vstack((np.arange(book_num), np.arange(book_num)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(book_num))
        features = th.sparse.FloatTensor(indices, values, th.Size([book_num, book_num]))
    # author labels

    labels = dl.labels_test['data'][:book_num] + dl.labels_train['data'][:book_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 7

    train_valid_mask = dl.labels_train['mask'][:book_num]
    test_mask = dl.labels_test['mask'][:book_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['00', '00'], ['01', '10'], ['05', '52', '20'], ['04', '40'], ['04', '43', '30'], ['06', '61', '10'],
                  ['07', '70'], ]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_dblp(feat_type=0):
    # prefix = '../../data/DBLP'
    dl=data_loader(os.path.join(os.path.dirname(__file__),'DBLP'))
    # link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}
    link_type_dic={0:'author-paper',1:'paper-term',2:'paper-venue',3:'paper-author',4:'term-paper',5:'venue-paper'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        # src_type = str(dl.links['meta'][link_type][0])
        # dst_type = str(dl.links['meta'][link_type][1])
        if link_type==0:
            src_type='author'
            dst_type='paper'
        elif link_type==1:
            src_type='paper'
            dst_type='term'
        elif link_type==2:
            src_type='paper'
            dst_type='venue'
        elif link_type==3:
            src_type='paper'
            dst_type='author'
        elif link_type==4:
            src_type='term'
            dst_type='paper'
        elif link_type==5:
            src_type='venue'
            dst_type='paper'

        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        if link_type==0:
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-4057)
        elif link_type==1:
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-4057,dl.links['data'][link_type].nonzero()[1]-18385)
        elif link_type==2:
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-4057,dl.links['data'][link_type].nonzero()[1]-26108)
        elif link_type==3:
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-4057,dl.links['data'][link_type].nonzero()[1])
        elif link_type==4:
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-18385,dl.links['data'][link_type].nonzero()[1]-4057)
        else:
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-26108,dl.links['data'][link_type].nonzero()[1]-4057)

    hg = dgl.heterograph(data_dic)

    hg.nodes['author'].data['test_mask']=torch.tensor(dl.labels_test['mask'][:author_num],dtype=torch.uint8)
    hg.nodes['author'].data['train_mask']=torch.tensor(dl.labels_train['mask'][:author_num],dtype=torch.uint8)
    hg.nodes['author'].data['h']=torch.tensor(dl.nodes['attr'][0]).to(torch.float32)
    hg.nodes['paper'].data['h']=torch.tensor(dl.nodes['attr'][1]).to(torch.float32)
    hg.nodes['term'].data['h']=torch.tensor(dl.nodes['attr'][2]).to(torch.float32) 
    meta_paths_dict = {
                        'APA': [('author', 'author-paper', 'paper'),
                                ('paper', 'paper-author', 'author')],
                        'APTPA': [('author', 'author-paper', 'paper'), 
                                  ('paper', 'paper-term', 'term'),
                                  ('term', 'term-paper', 'paper'), 
                                  ('paper', 'paper-author', 'author')],
                        'APVPA': [('author', 'author-paper', 'paper'), 
                                  ('paper', 'paper-venue', 'venue'),
                                  ('venue', 'venue-paper', 'paper'), 
                                  ('paper', 'paper-author', 'author')],
                      }

    graph_data={}
    for key, mp in meta_paths_dict.items():
        mp_g = dgl.metapath_reachable_graph(hg, mp)
        n_edge = mp_g.canonical_etypes[0]
        graph_data[(n_edge[0], key, n_edge[2])] = mp_g.edges()
    new_hg=dgl.heterograph(graph_data)
    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(author_num))

    # author labels

    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)
    labels=labels.to(torch.float32)
    hg.nodes['author'].data['label']=labels
    return hg 
    num_classes = 4

    train_valid_mask = dl.labels_train['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths

def load_Freebase3(feat_type=0):
    dl=data_loader(os.path.join(os.path.dirname(__file__),'Freebase3'))
    link_type_dic={
        0:'movie->actor',1:'actor->movie',2:'movie->direct',3:'direct->movie',4:'movie->writer',5:'writer->movie'
    }
    movie_num=dl.nodes['count'][0]
    data_dic={}
    for link_type in dl.links['data'].keys():
        if link_type==0:
            src_type='movie'
            dst_type='actor'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-3492)
        elif link_type==1:
            src_type='actor'
            dst_type='movie'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0]-3492,dl.links['data'][link_type].nonzero()[1])
        elif link_type==2:
            src_type='movie'
            dst_type='direct'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-3492-33401)
        elif link_type==3:
            src_type='direct'
            dst_type='movie'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0]-3492-33401,dl.links['data'][link_type].nonzero()[1])
        elif link_type==4:
            src_type='movie'
            dst_type='writer'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-3492-33401-2502)
        elif link_type==5:
            src_type='writer'
            dst_type='movie'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0]-3492-33401-2502,dl.links['data'][link_type].nonzero()[1])
    hg=dgl.heterograph(data_dic)
    hg.nodes['movie'].data['test_mask']=torch.tensor(dl.labels_test['mask'][:movie_num],dtype=torch.uint8)
    hg.nodes['movie'].data['train_mask']=torch.tensor(dl.labels_train['mask'][:movie_num],dtype=torch.uint8)
    indices = np.vstack((np.arange(dl.nodes['count'][0]), np.arange(dl.nodes['count'][0])))
    indices = torch.LongTensor(indices)
    values = torch.tensor(np.ones(dl.nodes['count'][0])).to(torch.float32)
    hg.nodes['movie'].data['h']= torch.sparse.FloatTensor(indices, values, torch.Size([dl.nodes['count'][0], dl.nodes['count'][0]]))
    indices = np.vstack((np.arange(dl.nodes['count'][1]), np.arange(dl.nodes['count'][1])))
    indices = torch.LongTensor(indices)
    values = torch.tensor(np.ones(dl.nodes['count'][1])).to(torch.float32)
    hg.nodes['actor'].data['h']=torch.sparse.FloatTensor(indices, values, torch.Size([dl.nodes['count'][1], dl.nodes['count'][1]]))
    indices = np.vstack((np.arange(dl.nodes['count'][2]), np.arange(dl.nodes['count'][2])))
    indices = torch.LongTensor(indices)
    values = torch.tensor(np.ones(dl.nodes['count'][2])).to(torch.float32)
    hg.nodes['direct'].data['h']=torch.sparse.FloatTensor(indices, values, torch.Size([dl.nodes['count'][2], dl.nodes['count'][2]]))
    # hg.nodes['keyword'].data['h']=torch.tensor(dl.nodes['attr'][3]).to(torch.float32) 
    indices = np.vstack((np.arange(dl.nodes['count'][3]), np.arange(dl.nodes['count'][3])))
    indices = torch.LongTensor(indices)
    values = torch.tensor(np.ones(dl.nodes['count'][3])).to(torch.float32)
    hg.nodes['writer'].data['h']=torch.sparse.FloatTensor(indices, values, torch.Size([dl.nodes['count'][3], dl.nodes['count'][3]]))
    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels=torch.tensor(labels)
    hg.nodes['movie'].data['label']=labels.argmax(axis=1)
    return hg


def load_imdb(feat_type=0):
    dl=data_loader(os.path.join(os.path.dirname(__file__),'IMDB'))
    # link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    link_type_dic={0:'movie->director',1:'director->movie',2:'movie->actor',3:'actor->movie',
                   4:'movie->keyword',5:'keyword->movie'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        if link_type==0:
            src_type='movie'
            dst_type='director'
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-4932)

        elif link_type==1:
            src_type='director'
            dst_type='movie'
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-4932,dl.links['data'][link_type].nonzero()[1])

        elif link_type==2:
            src_type='movie'
            dst_type='actor'
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-7325)

        elif link_type==3:
            src_type='actor'
            dst_type='movie'
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-7325,dl.links['data'][link_type].nonzero()[1])

        elif link_type==4:
            src_type='movie'
            dst_type='keyword'
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-13449)

        elif link_type==5:
            src_type='keyword'
            dst_type='movie'
            data_dic[(src_type, link_type_dic[link_type], dst_type)] = (dl.links['data'][link_type].nonzero()[0]-13499,dl.links['data'][link_type].nonzero()[1])

        '''
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        '''
    hg = dgl.heterograph(data_dic)
    hg.nodes['movie'].data['test_mask']=torch.tensor(dl.labels_test['mask'][:movie_num],dtype=torch.uint8)
    hg.nodes['movie'].data['train_mask']=torch.tensor(dl.labels_train['mask'][:movie_num],dtype=torch.uint8)
    hg.nodes['movie'].data['h']=torch.tensor(dl.nodes['attr'][0]).to(torch.float32)
    hg.nodes['director'].data['h']=torch.tensor(dl.nodes['attr'][1]).to(torch.float32)
    hg.nodes['actor'].data['h']=torch.tensor(dl.nodes['attr'][2]).to(torch.float32) 
    # hg.nodes['keyword'].data['h']=torch.tensor(dl.nodes['attr'][3]).to(torch.float32) 
    
    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels=th.tensor(labels)
    # labels = th.FloatTensor(labels)
    hg.nodes['movie'].data['label']=labels
    return hg 
    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(movie_num))

    # author labels

    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = th.FloatTensor(labels)

    num_classes = 5

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    # meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    meta_paths=[['movie->director','director->movie'],['movie->actor','actor->movie'],['movie->keyword','keyword->movie']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, dl
    

def load_data(dataset, feat_type=0):
    load_fun = None
    if dataset == 'ACM':
        load_fun = load_acm
    elif dataset == 'Freebase':
        feat_type = 1
        load_fun = load_freebase
    elif dataset == 'DBLP':
        load_fun = load_dblp
    elif dataset == 'IMDB':
        load_fun = load_imdb
    return load_fun(feat_type=feat_type)


def load_AMiner(feat_type=0):
    dl=data_loader(os.path.join(os.path.dirname(__file__),'AMiner'))
    link_type_dic={0:'paper->author',1:'author->paper',2:'paper->reference',3:'reference->paper'}
    paper_num=dl.nodes['count'][0]
    data_dic={}
    for link_type in dl.links['data'].keys():
        if link_type==0:
            src_type='paper'
            dst_type='author'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-6564)
        elif link_type==1:
            src_type='author'
            dst_type='paper'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0]-6564,dl.links['data'][link_type].nonzero()[1])
        elif link_type==2:
            src_type='paper'
            dst_type='reference'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0],dl.links['data'][link_type].nonzero()[1]-6564-13329)
        elif link_type==3:
            src_type='reference'
            dst_type='paper'
            data_dic[(src_type,link_type_dic[link_type],dst_type)]=(dl.links['data'][link_type].nonzero()[0]-6564-13329,dl.links['data'][link_type].nonzero()[1])
    hg=dgl.heterograph(data_dic)
    hg.nodes['paper'].data['test_mask']=torch.tensor(dl.labels_test['mask'][:paper_num],dtype=torch.uint8)
    hg.nodes['paper'].data['train_mask']=torch.tensor(dl.labels_train['mask'][:paper_num],dtype=torch.uint8)
    indices = np.vstack((np.arange(dl.nodes['count'][0]), np.arange(dl.nodes['count'][0])))
    indices = torch.LongTensor(indices)
    values = torch.tensor(np.ones(dl.nodes['count'][0])).to(torch.float32)
    hg.nodes['paper'].data['h']= torch.sparse.FloatTensor(indices, values, torch.Size([dl.nodes['count'][0], dl.nodes['count'][0]]))
    indices = np.vstack((np.arange(dl.nodes['count'][1]), np.arange(dl.nodes['count'][1])))
    indices = torch.LongTensor(indices)
    values = torch.tensor(np.ones(dl.nodes['count'][1])).to(torch.float32)
    hg.nodes['author'].data['h']=torch.sparse.FloatTensor(indices, values, torch.Size([dl.nodes['count'][1], dl.nodes['count'][1]]))
    indices = np.vstack((np.arange(dl.nodes['count'][2]), np.arange(dl.nodes['count'][2])))
    indices = torch.LongTensor(indices)
    values = torch.tensor(np.ones(dl.nodes['count'][2])).to(torch.float32)
    hg.nodes['reference'].data['h']=torch.sparse.FloatTensor(indices, values, torch.Size([dl.nodes['count'][2], dl.nodes['count'][2]]))
    # hg.nodes['keyword'].data['h']=torch.tensor(dl.nodes['attr'][3]).to(torch.float32) 
    
    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels=torch.tensor(labels)
    hg.nodes['paper'].data['label']=labels.argmax(axis=1)
    return hg 


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc <= self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

if __name__=='__main__':
    hg=load_acm()
    # hg=load_dblp()
    # hg=load_Freebase3()
    print(hg)