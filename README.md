The HG-Search method requires three stages: the first stage is pre search, the second stage is architecture search, and the third stage is hyperparameter search. 

In the pre search stage, it is necessary to preserve all network architecture space and hyperparameter space before conducting the search. The specific approach is as follows:
1.Keep all options in the self.search_space structure in the 'Search/search_space.py' file
2.
To perform a pre search on  DBLP dataset, you should execute the 
```python Search/main.py --dataset HGBn-DBLP --dataset_name HGBn-DBLP --predictfile HGBn-DBLP```
To perform  pre search on  IMDB dataset, you should execute the "python Search/main.py --dataset HGBn-IMDB --dataset_name HGBn-IMDB --predictfile HGBn-IMDB" command. 
To perform a pre search on  AMiner dataset, you should execute the "python Search/main.py --dataset HGBn-AMiner --dataset_name HGBn-AMiner --predictfile HGBn-AMiner" command.
To perform  pre search on  Freebase dataset, you should execute the "python Search/main.py --dataset HGBn-Freebase --dataset_name HGBn-Freebase --predictfile HGBn-Freebase" command.


In the architecture search phase, it is necessary to fix hyperparameters and then search based on the network architecture space. The specific approach is as follows:

1.Fix the hyperparameter related options in the "Search/search_space.py" file and preserve the schema search space.
2.
To perform a architecture search on  DBLP dataset, you should execute the "python Search/main.py --dataset HGBn-DBLP --dataset_name HGBn-DBLP --predictfile HGBn-DBLP" command. 
To perform  architecture search on  IMDB dataset, you should execute the "python Search/main.py --dataset HGBn-IMDB --dataset_name HGBn-IMDB --predictfile HGBn-IMDB" command. 
To perform a architecture search on  AMiner dataset, you should execute the "python Search/main.py --dataset HGBn-AMiner --dataset_name HGBn-AMiner --predictfile HGBn-AMiner" command.
To perform  architecture search on  Freebase dataset, you should execute the "python Search/main.py --dataset HGBn-Freebase --dataset_name HGBn-Freebase --predictfile HGBn-Freebase" command.

In the hyperparameter search stage, a fixed network architecture is required, and then the search is conducted based on the hyperparameter search space.The specific approach is as follows:

1.Fix the architecture  related options in the "Search/search_space.py" file and preserve the hyperparameter search space.
2.
To perform a hyperparameter  search on  DBLP dataset, you should execute the "python Search/main.py --dataset HGBn-DBLP --dataset_name HGBn-DBLP --predictfile HGBn-DBLP" command. 
To perform  hyperparameter  search on  IMDB dataset, you should execute the "python Search/main.py --dataset HGBn-IMDB --dataset_name HGBn-IMDB --predictfile HGBn-IMDB" command. 
To perform a hyperparameter  search on  AMiner dataset, you should execute the "python Search/main.py --dataset HGBn-AMiner --dataset_name HGBn-AMiner --predictfile HGBn-AMiner" command.
To perform  hyperparameter  search on  Freebase dataset, you should execute the "python Search/main.py --dataset HGBn-Freebase --dataset_name HGBn-Freebase --predictfile HGBn-Freebase" command.

The options related to architecture in the "Search/search_space.py" file are as follows:
model,subgraph_extraction,activation,gnn_type,has_bn,has_l2norm,macro_func,stage_type.

The options related to hyperparameter in the "Search/search_space.py" file are as follows:
dropout,feat,hidden_dim,layers_gnn,layers_post_mp,layers_pre_mp,lr,max_epoch,num_heads.


The HG-Search method requires a pre-search to be conducted first. Then, based on the optimal results obtained from the pre-search, the hyperparameters are fixed and the architecture search is performed. Next, based on the optimal results obtained from the architecture search, the architecture is fixed and the hyperparameter search is performed. To achieve excellent performance for different datasets, multiple iterations of architecture search and parameter search may be necessary.

