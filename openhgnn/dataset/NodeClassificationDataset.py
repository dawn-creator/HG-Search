from . import BaseDataset
from . import HGBDataset 
from ..utils import add_reverse_edges 
from . import register_dataset 
import torch

@register_dataset('node_classification')
class NodeClassificationDataset(BaseDataset):
    def __init__(self,*args,**kwargs):
        super(NodeClassificationDataset,self).__init__(*args,**kwargs)
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = False
        self.multi_label = False
        self.meta_paths_dict =None
    
    def get_labels(self):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels').long()
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label').long()
        else:
            raise ValueError('Labels of nodes are not in the hg.nodes[category].data.')
        labels = labels.float() if self.multi_label else labels
        return labels

    def get_split(self,validation=True):

        if 'train_mask' not in self.g.nodes[self.category].data:
            print("The dataset has no train mask. "
                  "So split the category nodes randomly. And the ratio of train/test is 8:2.")
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test
    
            train, test = torch.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = torch.tensor(train.indices)
            test_idx = torch.tensor(test.indices)
            if validation:
                print("Split train into train/valid with the ratio of 8:2 ")
                random_int = torch.randperm(len(train_idx))
                valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                print("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx 
        else:
            train_mask=self.g.nodes[self.category].data.pop('train_mask')
            test_mask=self.g.nodes[self.category].data.pop('test_mask')
            train_idx=torch.nonzero(train_mask,as_tuple=False).squeeze()
            test_idx=torch.nonzero(test_mask,as_tuple=False).squeeze()
            if validation:
                
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    valid_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
                elif 'valid_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('valid_mask').squeeze()
                    valid_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                    # RDF_NodeClassification has train_mask, no val_mask
                    print("Split train into train/valid with the ratio of 8:2 ")
                    random_int = torch.randperm(len(train_idx))
                    '''
                    path=os.path.join(os.path.dirname(__file__),"../../dataset_split.txt")
                    with open(path,"r") as file:
                        # file.write(str(random_int.numpy().tolist()))
                        lines=file.read().split(',')
                    new_random_int=[]
                    for i in range(len(lines)):
                        new_random_int.append(int(lines[i]))
                    random_int=th.LongTensor(np.array(new_random_int))
                    '''
                        
                    valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                    train_idx = train_idx[random_int[len(train_idx) // 5:]]
                        
                    # train_ratio=0.01
                    # val_ratio=0.2
                    # train_ratio=train_ratio/(1-val_ratio)
                    # train_idx=train_idx[:int(len(train_idx)*train_ratio)]
            else:
                    print("Set valid set with train set.")
                    valid_idx=train_idx
                    train_idx=train_idx 
        self.train_idx=train_idx
        self.valid_idx=valid_idx
        self.test_idx=test_idx
        return self.train_idx,self.valid_idx,self.test_idx

    
@register_dataset('HGBn_node_classification')
class HGB_NodeClassification(NodeClassificationDataset):

    def __init__(self,dataset_name,*args,**kwargs):
        super(HGB_NodeClassification, self).__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.has_feature = True
        if dataset_name == 'HGBn-ACM':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'paper'
            num_classes = 3
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
            self.meta_paths_dict = {'PAP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')],
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
        elif dataset_name == 'HGBn-DBLP':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'author'
            num_classes = 4
            self.meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    'APTPA': [('author', 'author-paper', 'paper'), ('paper', 'paper-term', 'term'),
                                              ('term', 'term-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    'APVPA': [('author', 'author-paper', 'paper'), ('paper', 'paper-venue', 'venue'),
                                              ('venue', 'venue-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    }
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        elif dataset_name == 'HGBn-Freebase':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'BOOK'
            num_classes = 7
            self.has_feature = False
            g = add_reverse_edges(g)
            self.meta_paths_dict = {'BB': [('BOOK', 'BOOK-and-BOOK', 'BOOK')],
                                    'BFB': [('BOOK', 'BOOK-to-FILM', 'FILM'), ('FILM', 'BOOK-to-FILM-rev', 'BOOK')],
                                    'BOFB': [('BOOK', 'BOOK-about-ORGANIZATION', 'ORGANIZATION'),
                                             ('ORGANIZATION', 'ORGANIZATION-in-FILM', 'FILM'),
                                             ('FILM', 'BOOK-to-FILM-rev', 'BOOK')],
                                    'BLMB': [('BOOK', 'BOOK-on-LOCATION', 'LOCATION'),
                                             ('LOCATION', 'MUSIC-on-LOCATION-rev', 'MUSIC'),
                                             ('MUSIC', 'MUSIC-in-BOOK', 'BOOK')],
                                    'BPB': [('BOOK', 'PEOPLE-to-BOOK-rev', 'PEOPLE'),
                                            ('PEOPLE', 'PEOPLE-to-BOOK', 'BOOK')],
                                    'BPSB': [('BOOK', 'PEOPLE-to-BOOK-rev', 'PEOPLE'),
                                             ('PEOPLE', 'PEOPLE-to-SPORTS', 'SPORTS'),
                                             ('SPORTS', 'BOOK-on-SPORTS-rev', 'BOOK')],
                                    'BBuB': [('BOOK', 'BUSINESS-about-BOOK-rev', 'BUSINESS'),
                                             ('BUSINESS', 'BUSINESS-about-BOOK', 'BOOK')],
                                    # 'BOMB': [('BOOK', 'BOOK-about-ORGANIZATION', 'ORGANIZATION'),
                                    #          ('ORGANIZATION', 'ORGANIZATION-to-MUSIC', 'MUSIC'),
                                    #          ('MUSIC', 'MUSIC-in-BOOK', 'BOOK')],
                                    # 'BOBuB': [('BOOK', 'BOOK-about-ORGANIZATION', 'ORGANIZATION'),
                                    #           ('ORGANIZATION', 'ORGANIZATION-for-BUSINESS', 'BUSINESS'),
                                    #           ('BUSINESS', 'BUSINESS-about-BOOK', 'BOOK')]
                                    }
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        elif dataset_name=='HGBn-AMiner':
            dataset=HGBDataset(name=dataset_name,raw_dir='')
            g=dataset[0].long()
            category='paper'
            num_classes=4
            self.meta_paths_dict={
                'PAP':[('paper','paper->author','author'),('author','author->paper','paper')],
                'PRP':[('paper','paper->reference','reference'),('reference','reference->paper','paper')]
            }
        elif dataset_name == 'HGBn-IMDB':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'movie'
            num_classes = 5
            self.meta_paths_dict = {
                'MAM': [('movie', 'movie->actor', 'actor'), ('actor', 'actor->movie', 'movie')],
                'MDM': [('movie', 'movie->director', 'director'), ('director', 'director->movie', 'movie')],
                'MKM': [('movie', 'movie->keyword', 'keyword'), ('keyword', 'keyword->movie', 'movie')],
                'DMD': [('director', 'director->movie', 'movie'), ('movie', 'movie->director', 'director')],
                'DMAMD': [('director', 'director->movie', 'movie'), ('movie', 'movie->actor', 'actor'),
                          ('actor', 'actor->movie', 'movie'), ('movie', 'movie->director', 'director')],
                'AMA': [('actor', 'actor->movie', 'movie'), ('movie', 'movie->actor', 'actor')],
                'AMDMA': [('actor', 'actor->movie', 'movie'), ('movie', 'movie->director', 'director'),
                          ('director', 'director->movie', 'movie'), ('movie', 'movie->actor', 'actor')]
            }
            # RuntimeError: result type Float can't be cast to the desired output type Long
            self.multi_label = True
        elif dataset_name=='HGBn-Freebase3':
            dataset=HGBDataset(name=dataset_name,raw_dir='')
            g=dataset[0].long()
            category='movie'
            num_classes=3
            
            self.meta_paths_dict={
                'MAM':[('movie','movie->actor','actor'),('actor','actor->movie','movie')],
                'MDM':[('movie','movie->direct','direct'),('direct','direct->movie','movie')],
                'MWM':[('movie','movie->writer','writer'),('writer','writer->movie','movie')],
                'DMD':[('direct','direct->movie','movie'),('movie','movie->direct','direct')],
                'DMAMD':[('direct','direct->movie','movie'),('movie','movie->actor','actor'),('actor','actor->movie','movie'),('movie','movie->direct','direct')],
                'AMA':[('actor','actor->movie','movie'),('movie','movie->actor','actor')],
                'AMDMA':[('actor','actor->movie','movie'),('movie','movie->direct','direct'),('direct','direct->movie','movie'),('movie','movie->actor','actor')]
            }
            
            '''
            self.meta_paths_dict={
                'MAM':[('movie','movie->actor','actor'),('actor','actor->movie','movie')],
                'MDM':[('movie','movie->direct','direct'),('direct','direct->movie','movie')],
                'MWM':[('movie','movie->writer','writer'),('writer','writer->movie','movie')]
            }
            '''
        else:
            raise ValueError
        self.g, self.category, self.num_classes = g, category, num_classes