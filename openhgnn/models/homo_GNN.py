import dgl 
from ..layers import SkipConnection 
from . import register_model 
from .HeteroMLP import HGNNPostMP,HGNNPreMP
from . import BaseModel 

stage_dict={
    'stack':SkipConnection.GNNStackStage,
    'skipsum':SkipConnection.GNNSkipStage,
    'skipconcat':SkipConnection.GNNSkipStage,
}


@register_model('homo_GNN')
class homo_GNN(BaseModel):
   
    @classmethod
    def build_model_from_args(cls, args, hg):
        out_node_type = args.out_node_type
       
        
        #DBLP
        if args.dataset=="HGBn-DBLP":
            if args.author_paper==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="author-paper")[2],"author-paper")
            if args.paper_author==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="paper-author")[2],"paper-author")
            if args.paper_term==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="paper-term")[2],"paper-term")
            if args.term_paper==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="term-paper")[2],"term-paper")
            if args.paper_venue==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="paper-venue")[2],"paper-venue")
            if args.venue_paper==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="venue-paper")[2],"venue-paper")
            
        # IMDB
        if args.dataset=="HGBn-IMDB":
            if args.actor_movie==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="actor->movie")[2],"actor->movie")
            if args.movie_actor==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="movie->actor")[2],"movie->actor")
            if args.director_movie==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="director->movie")[2],"director->movie")
            if args.movie_director==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="movie->director")[2],"movie->director")
            if args.keyword_movie==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="keyword->movie")[2],"keyword->movie")
            if args.movie_keyword==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="movie->keyword")[2],"movie->keyword")
            
        # Freebase3
        if args.dataset=="HGBn-Freebase3":
            if args.movie_actor==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="movie->actor")[2],"movie->actor")
            if args.actor_movie==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="actor->movie")[2],"actor->movie")
            if args.movie_direct==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="movie->direct")[2],"movie->direct")
            if args.direct_movie==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="direct->movie")[2],"direct->movie")
            if args.movie_writer==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="movie->writer")[2],"movie->writer")
            if args.writer_movie==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="writer->movie")[2],"writer->movie")   
            

        # AMiner
        if args.dataset=="HGBn-AMiner":
            if args.paper_author==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="paper->author")[2],"paper->author")
            if args.author_paper==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="author->paper")[2],"author->paper")
            if args.paper_reference==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="paper->reference")[2],"paper->reference")
            if args.reference_paper==0:
                hg=dgl.remove_edges(hg,hg.edges("all",etype="reference->paper")[2],"reference->paper")

        return cls(args, hg, out_node_type)

    def __init__(self, args, hg, out_node_type, **kwargs):
        super(homo_GNN, self).__init__()
        self.hg=hg
        self.out_node_type = out_node_type
        if args.layers_pre_mp - 1 > 0:
            self.pre_mp = HGNNPreMP(args, hg.ntypes, args.layers_pre_mp, args.hidden_dim, args.hidden_dim)
        if args.layers_gnn > 0:
            GNNStage = stage_dict[args.stage_type]
            self.gnn = GNNStage(gnn_type=args.gnn_type,
                                stage_type=args.stage_type,
                                dim_in=args.hidden_dim,
                                dim_out=args.hidden_dim,
                                num_layers=args.layers_gnn,
                                skip_every=1,
                                dropout=args.dropout,
                                act=args.activation,
                                has_bn=args.has_bn,
                                num_heads=args.num_heads,
                                has_l2norm=args.has_l2norm,
                                num_etypes=len(hg.etypes),
                                num_ntypes=len(hg.ntypes))

        gnn_out_dim = self.gnn.dim_out
        self.post_mp = HGNNPostMP(args, self.out_node_type, args.layers_post_mp, gnn_out_dim, args.out_dim)

    def forward(self, hg, h_dict):
        with hg.local_scope():
            hg=self.hg 
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if len(hg.ntypes) == 1:
                hg.ndata['h'] = h_dict[hg.ntypes[0]]
            else:
                hg.ndata['h'] = h_dict
            homo_g = dgl.to_homogeneous(hg, ndata=['h'])
            homo_g = dgl.remove_self_loop(homo_g)
            homo_g = dgl.add_self_loop(homo_g)
            h = homo_g.ndata.pop('h')
            if hasattr(self, 'gnn'):
                h = self.gnn(homo_g, h)
                if len(hg.ntypes) == 1:
                    out_h = {hg.ntypes[0]: h}
                else:
                    out_h = self.h2dict(h, hg.ndata['h'], self.out_node_type)
            if hasattr(self, 'post_mp'):
                out_h = self.post_mp(out_h)
        return out_h

    def h2dict(self, h, hdict, node_list):
        pre = 0
        out_h = {}
        for i, value in hdict.items():
            if i in node_list:
                out_h[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return out_h