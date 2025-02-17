# The HG-Search method requires three stages: 
pre search stage, architecture search stage, hyperparameter search stage. 


## To perform a pre search stage on  DBLP/IMDB/AMiner/Freebase dataset 
You should keep all options in the self.search_space structure in the 'Search/search_space.py' file and execute the following command,

```
   python Search/main.py --dataset HGBn-DBLP --dataset_name HGBn-DBLP --predictfile HGBn-DBLP
   python Search/main.py --dataset HGBn-IMDB --dataset_name HGBn-IMDB --predictfile HGBn-IMDB
   python Search/main.py --dataset HGBn-AMiner --dataset_name HGBn-AMiner --predictfile HGBn-AMiner
   python Search/main.py --dataset HGBn-Freebase --dataset_name HGBn-Freebase --predictfile HGBn-Freebase
```
## To perform the architecture search stage on  DBLP/IMDB/AMiner/Freebase dataset 
You should fix the hyperparameter related options in the "Search/search_space.py" file and preserve the schema search space, then execute the following command,
```
   python Search/main.py --dataset HGBn-DBLP --dataset_name HGBn-DBLP --predictfile HGBn-DBLP
   python Search/main.py --dataset HGBn-IMDB --dataset_name HGBn-IMDB --predictfile HGBn-IMDB
   python Search/main.py --dataset HGBn-AMiner --dataset_name HGBn-AMiner --predictfile HGBn-AMiner
   python Search/main.py --dataset HGBn-Freebase --dataset_name HGBn-Freebase --predictfile HGBn-Freebase
```

## To perform the hyperparameter search stage on DBLP/IMDB/AMiner/Freebase dataset

In the hyperparameter search stage, a fixed network architecture is required, and then the search is conducted based on the hyperparameter search space. You should fix the architecture related options in the "Search/search_space.py" file and preserve the hyperparameter search space.
```
   python Search/main.py --dataset HGBn-DBLP --dataset_name HGBn-DBLP --predictfile HGBn-DBLP
   python Search/main.py --dataset HGBn-IMDB --dataset_name HGBn-IMDB --predictfile HGBn-IMDB
   python Search/main.py --dataset HGBn-AMiner --dataset_name HGBn-AMiner --predictfile HGBn-AMiner
   python Search/main.py --dataset HGBn-Freebase --dataset_name HGBn-Freebase --predictfile HGBn-Freebase
```

The options related to architecture in the "Search/search_space.py" file are as follows: model,subgraph_extraction,activation,gnn_type,has_bn,has_l2norm,macro_func,stage_type.

The options related to hyperparameter in the "Search/search_space.py" file are as follows: dropout,feat,hidden_dim,layers_gnn,layers_post_mp,layers_pre_mp,lr,max_epoch,num_heads.


The HG-Search method requires a pre-search to be conducted first. Then, based on the optimal results obtained from the pre-search, the hyperparameters are fixed and the architecture search is performed. Next, based on the optimal results obtained from the architecture search, the architecture is fixed and the hyperparameter search is performed. To achieve excellent performance for different datasets, multiple iterations of architecture search and parameter search may be necessary.

