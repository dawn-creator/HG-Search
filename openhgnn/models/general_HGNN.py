import dgl 
from ..layers import SkipConnection 
from openhgnn.models import BaseModel,register_model 
from .HeteroMLP import HGNNPostMP,HGNNPreMP

stage_dict={
    'stack':SkipConnection.HGNNStackStage,
    'skipsum':SkipConnection.HGNNSkipStage,
    'skipconcat':SkipConnection.HGNNSkipStage,
}

def HG_transformation(hg,metapaths_dict):
    graph_data={}
    for key, mp in metapaths_dict.items():
        mp_g = dgl.metapath_reachable_graph(hg, mp)
        n_edge = mp_g.canonical_etypes[0]
        graph_data[(n_edge[0], key, n_edge[2])] = mp_g.edges()
    return dgl.heterograph(graph_data)


@register_model('general_HGNN')
class general_HGNN(BaseModel):

    @classmethod
    def build_model_from_args(cls,args,hg):
        out_node_type=args.out_node_type 
        if args.subgraph_extraction=='relation':
            new_hg=hg
            print("relation extraction!")
            #DBLP
            if args.dataset=="HGBn-DBLP":
                if args.author_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="author-paper")[2],"author-paper")
                if args.paper_author==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper-author")[2],"paper-author")
                if args.paper_term==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper-term")[2],"paper-term")
                if args.term_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="term-paper")[2],"term-paper")
                if args.paper_venue==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper-venue")[2],"paper-venue")
                if args.venue_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="venue-paper")[2],"venue-paper")
            
            # IMDB
            if args.dataset=="HGBn-IMDB":
                if args.actor_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="actor->movie")[2],"actor->movie")
                if args.movie_actor==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->actor")[2],"movie->actor")
                if args.director_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="director->movie")[2],"director->movie")
                if args.movie_director==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->director")[2],"movie->director")
                if args.keyword_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="keyword->movie")[2],"keyword->movie")
                if args.movie_keyword==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->keyword")[2],"movie->keyword")
            
            # Freebase3
            if args.dataset=="HGBn-Freebase3":
                if args.movie_actor==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->actor")[2],"movie->actor")
                if args.actor_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="actor->movie")[2],"actor->movie")
                if args.movie_direct==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->direct")[2],"movie->direct")
                if args.direct_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="direct->movie")[2],"direct->movie")
                if args.movie_writer==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->writer")[2],"movie->writer")
                if args.writer_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="writer->movie")[2],"writer->movie")   
            

            # AMiner
            if args.dataset=="HGBn-AMiner":
                if args.paper_author==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper->author")[2],"paper->author")
                if args.author_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="author->paper")[2],"author->paper")
                if args.paper_reference==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper->reference")[2],"paper->reference")
                if args.reference_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="reference->paper")[2],"reference->paper")
                
        elif args.subgraph_extraction=='metapath':
            if hasattr(args,'meta_paths_dict'):
                new_hg=HG_transformation(hg,args.meta_paths_dict)
                print("metapath extraction!")
            else:
                raise ValueError('No meta-path is specified!') 
        elif args.subgraph_extraction=='mixed':
            relation_dict=args.meta_paths_dict
            for etype in hg.canonical_etypes:
                relation_dict[etype[1]] = [etype]
            new_hg = HG_transformation(hg, relation_dict)
            print('mixed extraction!')
            
            #DBLP
            if args.dataset=="HGBn-DBLP":
                if args.author_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="author-paper")[2],"author-paper")
                if args.paper_author==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper-author")[2],"paper-author")
                if args.paper_term==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper-term")[2],"paper-term")
                if args.term_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="term-paper")[2],"term-paper")
                if args.paper_venue==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper-venue")[2],"paper-venue")
                if args.venue_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="venue-paper")[2],"venue-paper")
            
            # IMDB
            if args.dataset=="HGBn-IMDB":
                if args.actor_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="actor->movie")[2],"actor->movie")
                if args.movie_actor==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->actor")[2],"movie->actor")
                if args.director_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="director->movie")[2],"director->movie")
                if args.movie_director==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->director")[2],"movie->director")
                if args.keyword_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="keyword->movie")[2],"keyword->movie")
                if args.movie_keyword==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->keyword")[2],"movie->keyword")
            
            # Freebase3
            if args.dataset=="HGBn-Freebase3":
                if args.movie_actor==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->actor")[2],"movie->actor")
                if args.actor_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="actor->movie")[2],"actor->movie")
                if args.movie_direct==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->direct")[2],"movie->direct")
                if args.direct_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="direct->movie")[2],"direct->movie")
                if args.movie_writer==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="movie->writer")[2],"movie->writer")
                if args.writer_movie==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="writer->movie")[2],"writer->movie")   
            

            # AMiner
            if args.dataset=="HGBn-AMiner":
                if args.paper_author==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper->author")[2],"paper->author")
                if args.author_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="author->paper")[2],"author->paper")
                if args.paper_reference==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="paper->reference")[2],"paper->reference")
                if args.reference_paper==0:
                    new_hg=dgl.remove_edges(new_hg,new_hg.edges("all",etype="reference->paper")[2],"reference->paper")
                
        else:
            raise ValueError('subgraph_extraction only supports relation, metapath and mixed')
        
        return cls(args,new_hg,out_node_type)
    
    def __init__(self,args,hg,out_node_type,**kwargs):

        super(general_HGNN,self).__init__()
        self.hg=hg
        self.out_node_type=out_node_type

        if args.layers_pre_mp-1>0:
            self.pre_mp=HGNNPreMP(args,self.hg.ntypes,args.layers_pre_mp,args.hidden_dim,args.hidden_dim)
        
        if args.layers_gnn > 0:
            HGNNStage = stage_dict[args.stage_type]
            self.hgnn = HGNNStage(gnn_type=args.gnn_type,
                                  rel_names=self.hg.etypes,
                                  stage_type=args.stage_type,
                                  dim_in=args.hidden_dim,
                                  dim_out=args.hidden_dim,
                                  num_layers=args.layers_gnn,
                                  skip_every=1,
                                  dropout=args.dropout,
                                  act=args.activation,
                                  has_bn=args.has_bn,
                                  has_l2norm=args.has_l2norm,
                                  num_heads=args.num_heads,
                                  macro_func=args.macro_func)
        gnn_out_dim = self.hgnn.dim_out
        self.post_mp = HGNNPostMP(args, self.out_node_type, args.layers_post_mp, gnn_out_dim, args.out_dim)

    def forward(self,hg,h_dict):
        with hg.local_scope():
            hg = self.hg
            h_dict = {key: value for key, value in h_dict.items() if key in hg.ntypes}
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if hasattr(self, 'hgnn'):
                h_dict = self.hgnn(hg, h_dict)
            if hasattr(self, 'post_mp'):
                out_h = {}
                for key, value in h_dict.items():
                    if key in self.out_node_type:
                        out_h[key] = value
                out_h = self.post_mp(out_h)
        return out_h