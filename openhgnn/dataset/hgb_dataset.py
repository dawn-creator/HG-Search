import os 
import numpy as np 
from dgl.data.utils import download,extract_archive 
from dgl.data import DGLDataset 
from dgl.data.utils import load_graphs 
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__),"../../Dataset_process"))
from utils_ import load_imdb
from utils_ import load_dblp 
from utils_ import load_acm 
from utils_ import load_Freebase3
from utils_ import load_AMiner 
from dgl.data import DGLDataset 
class HGBDataset(DGLDataset):


    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'

    def __init__(self,name,raw_dir=None,force_reload=False,verbose=True):
        assert name in ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB','HGBn-Freebase3','HGBn-AMiner',
                        'HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']
        self.prefix_task = name[:4]
        self.dataset_name=name[5:]
        self.data_path = './openhgnn/dataset/{}.zip'.format(self.prefix_task)
        self.g_path = './openhgnn/dataset/{}/{}.bin'.format(self.prefix_task, name)
        raw_dir = './openhgnn/dataset'
        url = self._prefix + 'dataset/{}.zip'.format(self.prefix_task)
        super(HGBDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
    
    def download(self):
        if os.path.exists(self.data_path):
            pass
        else:
            file_path=os.path.join(self.raw_dir)
            download(self.url,path=file_path)
        extract_archive(self.data_path,os.path.join(self.raw_dir,self.prefix_task))

    def process(self):
        if self.dataset_name=="DBLP":
            self._g=load_dblp()
        elif self.dataset_name=="IMDB":
            g, _ = load_graphs(self.g_path)
            self._g = g[0]
            self._g=load_imdb()
            self.judge(g[0],self._g)
            self._g=g[0]
        elif self.dataset_name=="Freebase3":
            self._g=load_Freebase3()
        elif self.dataset_name=="AMiner":
            self._g=load_AMiner()
        
        
        
        

    def judge(self,x_1,x_2):
        
        x_1.ndata['label']['movie']=x_2.ndata['label']['movie']
        x,y=x_1.ndata['label']['movie'].shape
        for num_1 in range(x):
            for num_2 in range(y):
                x_1.ndata['label']['movie'][num_1][num_2]=x_2.ndata['label']['movie'][num_1][num_2]
        return x_1 
        
        
    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass