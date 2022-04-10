
from typing import Dict
import torch as th
import torch.nn as nn
import torch.nn.functional as TFn
from dgl.utils import expand_as_pair
import dgl.function as fn
import dgl
import dgl.nn.pytorch as gnn
from dgl.nn.functional import edge_softmax
import math
import dgl.ops as F
from skip_edge_gnn import HeteroGraphConv


class IAGNN(nn.Module):
    '''
    Intention Adaptive Graph Neural Network
    ----
    try to introduce the position embedding on edge v2i

    Original 4 types of links (CDS Graph):\n
        1. user-item in Domain A.
        2. user-item in Domain B.
        3. seq items in Domain A.
        4. seq items in Domain B.
    '''
    def __init__(self,
                 num_class,
                 embedding_dim,
                 num_layers,
                 device,
                 batch_norm=True,
                 add_loss=False,
                 feat_drop=0.0,
                 attention_drop=0.0,
                 tao=1.0,
                 vinitial_type='mean',
                 graph_feature_select='gated',
                 pooling_type='last',
                 predictor_type='matmul'):
        super(IAGNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.aux_factor = 2  # hyper-parameter for aux information size
        self.auxemb_dim = int(self.embedding_dim // self.aux_factor)
        self.item_embedding = nn.Embedding(num_class['item'],
                                           embedding_dim,
                                           max_norm=1)
        self.cate_embedding = nn.Embedding(num_class['cate'],
                                           embedding_dim,
                                           max_norm=1)
        self.pos_embedding = nn.Embedding(num_class['pos'], self.auxemb_dim)

        self.num_layers = num_layers  # hyper-parameter for gnn layers
        self.add_loss = add_loss
        self.batch_norm = nn.BatchNorm1d(embedding_dim *
                                         2) if batch_norm else None

        self.readout = AttnReadout(
            embedding_dim,
            self.auxemb_dim,
            embedding_dim,
            pooling_type=pooling_type,
            tao=tao,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(embedding_dim),
        )
        self.finalfeature = FeatureSelect(embedding_dim, type=graph_feature_select)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                HeteroGraphConv({
                    # 'e':
                    # GATConv(embedding_dim,
                    #         embedding_dim,
                    #         feat_drop=feat_drop,
                    #         attn_drop=attention_drop),
                    # 'e2':
                    # GATConv(embedding_dim,
                    #         embedding_dim,
                    #         feat_drop=feat_drop,
                    #         attn_drop=attention_drop),
                    'i2i':
                    GATConv(embedding_dim,
                            embedding_dim,
                            feat_drop=feat_drop,
                            attn_drop=attention_drop),
                    'i2v':
                    GATConv(embedding_dim,
                            embedding_dim,
                            feat_drop=feat_drop,
                            attn_drop=attention_drop),
                    'v2v':
                    GATConv(embedding_dim,
                            embedding_dim,
                            feat_drop=feat_drop,
                            attn_drop=attention_drop),
                    'v2i':
                    GATConv(embedding_dim,
                            embedding_dim,
                            feat_drop=feat_drop,
                            attn_drop=attention_drop),
                    'c2c':
                    GATConv(embedding_dim,
                            embedding_dim,
                            feat_drop=feat_drop,
                            attn_drop=attention_drop),
                    'c2i':
                    GATConv(embedding_dim,
                            embedding_dim,
                            feat_drop=feat_drop,
                            attn_drop=attention_drop),
                    'i2c':
                    GATConv(embedding_dim,
                            embedding_dim,
                            feat_drop=feat_drop,
                            attn_drop=attention_drop),
                }))

        self.gnn_maxpooling_layer = HeteroGraphConv({
            'mp': MaxPoolingLayer(),
        })

        # W_h_e * (h_s || e_u) + b
        self.W_pos = nn.Parameter(
            th.Tensor(embedding_dim * 2 + self.auxemb_dim, embedding_dim))
        self.W_hs_e = nn.Parameter(th.Tensor(embedding_dim * 2, embedding_dim))
        self.W_h_e = nn.Parameter(th.Tensor(embedding_dim * 3, embedding_dim))
        self.W_c = nn.Parameter(
            th.Tensor(embedding_dim * 2, embedding_dim))
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)
        self.reset_parameters()
        self.indices = nn.Parameter(th.arange(num_class['item'],
                                              dtype=th.long),
                                    requires_grad=False)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def feature_encoder(self, g: dgl.DGLHeteroGraph, next_cid: th.Tensor):
        iid = g.nodes['i'].data['id']
        vid = g.nodes['v'].data['id']
        cid = g.nodes['c'].data['id']

        # store the embedding in graph
        g.update_all(fn.copy_e('pos', 'ft'),
                     fn.min('ft', 'f_pos'),
                     etype='v2i')
        pos_emb = self.pos_embedding(g.nodes['i'].data['f_pos'].long())
        cat_emb = th.cat([
            self.item_embedding(iid), pos_emb,
            self.cate_embedding(g.nodes['i'].data['cate'])
        ],
                         dim=1)
        g.nodes['i'].data['f'] = th.matmul(cat_emb, self.W_pos)
        g.nodes['v'].data['f'] = self.cate_embedding(vid)
        g.nodes['c'].data['f'] = self.cate_embedding(cid)
            # th.cat([self.cate_embedding(cid), pos_emb], dim=-1), self.W_c)

        return self.cate_embedding(next_cid)

    def forward(self, g: dgl.DGLHeteroGraph, next_cid: th.Tensor):
        '''
        Args:
        ----
            g (dgl.DGLHeteroGraph): a dgl.batch of HeteroGraphs
            next_cid (th.Tensor): a batch of next category ids [bs, 1]
        '''

        next_cate = self.feature_encoder(g, next_cid)

        # main multi-layer GNN
        h = [{
            'i': g.nodes['i'].data['f'],
            'v': g.nodes['v'].data['f'],
            'c': g.nodes['c'].data['f']
        }]  # a list feat record for every layers
        for i, layer in enumerate(self.gnn_layers):
            out = layer(g, (h[-1], h[-1]))
            h.append(out)

        # h[-1]['v']:                   [bs*1, 1, embsize]
        # h[-1]['i']:                   [items_len_in_bs, 1, embsize]
        # g.nodes['i'].data['cate']:    [items_len_in_bs]
        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1,
                                    ntype='i')  # index array
        last_cnodes = g.filter_nodes(lambda nodes: nodes.data['clast'] == 1, ntype='c')
        seq_last_nodes = g.filter_nodes(
            lambda nodes: nodes.data['seq_last'] == 1,
            ntype='i')  # index array
        seq_last_cnodes = g.filter_nodes(
            lambda nodes: nodes.data['seq_clast'] == 1,
            ntype='c')  # index array

        # get max of item feat in the category sequence
        # max_pooling_result = self.gnn_maxpooling_layer(g, (h[-1], h[-1]))   # [items_len_in_bs, 1, embsize]
        # h_s = max_pooling_result['i'][last_nodes].squeeze() # [bs, embsize]


        # try gated feat
        feat = self.finalfeature(h)

        # use last item feat in the category sequence
        h_c = feat['i'][last_nodes].squeeze()  # [bs, embsize]
        # also add seq last
        h_s = feat['i'][seq_last_nodes].squeeze()  # [bs, embsize]
        gate = th.sigmoid(th.matmul(th.cat((h_c, h_s), 1), self.W_hs_e))
        h_all = gate * h_c + (1 - gate) * h_s

        feat_last_cate = feat['c'][last_cnodes].squeeze()
        feat_seq_last_cate = feat['c'][seq_last_cnodes].squeeze()
        c_gate = th.sigmoid(th.matmul(th.cat((feat_last_cate, feat_seq_last_cate), 1), self.W_c))
        c_all = c_gate * feat_last_cate + (1 - c_gate) * feat_seq_last_cate

        feat_next_cate = feat['v'].squeeze()
        all_feat = th.matmul(th.cat((h_all, c_all, feat_next_cate), 1),
                             self.W_h_e)  # [bs, embsize]

        cand_items = self.item_embedding(self.indices)

        # cosine predictor
        scores1 = th.matmul(all_feat, cand_items.t())
        scores1 = scores1 / th.sqrt(th.sum(cand_items * cand_items,
                                           1)).unsqueeze(0).expand_as(scores1)


        return scores1, feat['v'], g.batch_num_nodes('i')


class MaxPoolingLayer(nn.Module):
    '''
    for edge type 'mp' (maxpooling), make a 'max pooling' update
    '''
    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, g: dgl.DGLHeteroGraph, feat: Dict):
        with g.local_scope():
            g.srcdata.update({'ft': feat[0]})
            g.update_all(fn.copy_u('ft', 'f_m'), fn.max('f_m', 'f_max'))

            return g.dstdata['f_max']


class V2I_models(nn.Module):
    def __init__(self,
                 in_dim: int,
                 aux_dim: int,
                 out_dim: int,
                 attn_drop: float = 0.1,
                 negative_slope: float = 0.2):
        super(V2I_models, self).__init__()

        self.W = nn.Linear(in_dim + aux_dim * 1, out_dim)
        self.W_extract_pos = nn.Linear(aux_dim * 1, out_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, g: dgl.DGLHeteroGraph, feat: Dict):
        srcdata = feat[0]
        dstdata = feat[1]
        with g.local_scope():
            g.srcdata.update({'ft': srcdata})
            g.dstdata.update({'ft': dstdata})

            e = self.leaky_relu(g.edata.pop('p'))
            # compute softmax
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))
            g.edata['a'] = self.W_extract_pos(g.edata['a'])
            # message passing
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = th.unsqueeze(g.dstdata['ft'], dim=1)

        return rst

class FeatureSelect(nn.Module):
    def __init__(self, embedding_dim, type='last'):
        super().__init__()
        self.embedding_dim = embedding_dim
        assert type in ['last', 'mean', 'gated']
        self.type = type

        self.W_g1 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.W_g2 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.W_g3 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

    def forward(self, h):
        h[0]['i'] = h[0]['i'].squeeze()
        h[-1]['i'] = h[-1]['i'].squeeze()
        h[0]['v'] = h[0]['v'].squeeze()
        h[-1]['v'] = h[-1]['v'].squeeze()
        h[0]['c'] = h[0]['c'].squeeze()
        h[-1]['c'] = h[-1]['c'].squeeze()
        feature = None
        if self.type == 'last':
            feature = h[-1]
        elif self.type == 'gated':
            gate = th.sigmoid(self.W_g1(th.cat([h[0]['i'], h[-1]['i']], dim=-1)))
            ifeature = gate * h[0]['i'] + (1 - gate) * h[-1]['i']

            gate = th.sigmoid(self.W_g2(th.cat([h[0]['v'], h[-1]['v']], dim=-1)))
            vfeature = gate * h[0]['v'] + (1 - gate) * h[-1]['v']

            gate = th.sigmoid(self.W_g3(th.cat([h[0]['c'], h[-1]['c']], dim=-1)))
            cfeature = gate * h[0]['c'] + (1 - gate) * h[-1]['c']

            feature = {'i': ifeature, 'v': vfeature, 'c': cfeature}
            # feature = {'i': ifeature, 'v': h[-1]['v'], 'c': h[-1]['c']}

        elif self.type == 'mean':
            isum = th.zeros_like(h[0]['i'])
            vsum = th.zeros_like(h[0]['v'])
            csum = th.zeros_like(h[0]['c'])
            for data in h:
                isum += data['i']
                vsum += data['v']
                csum += data['c']
            feature = {'i': isum / len(h), 'v': vsum / len(h), 'c': csum / len(h)}

        return feature

class AttnReadout(nn.Module):  # todo：需要对cross domain进行建模
    """
    Graph pooling for every session graph
    """
    def __init__(
        self,
        item_dim,
        aux_dim,
        output_dim,
        pooling_type='input',
        tao=1.0,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(item_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.w_feature = nn.Parameter(
            th.Tensor(item_dim + aux_dim * 1, output_dim))
        self.fc_u = nn.Linear(output_dim, output_dim, bias=False)
        self.fc_v = nn.Linear(output_dim, output_dim, bias=True)
        self.fc_e = nn.Linear(output_dim, 1, bias=False)
        self.fc_out = (nn.Linear(item_dim, output_dim, bias=False)
                       if output_dim != item_dim else None)
        self.activation = activation
        self.tao = tao
        assert pooling_type in ['ilast', 'imean', 'cmean', 'cnext', 'input']
        self.pooling_type = pooling_type

    def maxpooling_feat(self, g: dgl.DGLHeteroGraph, gfeat):
        pass

    # @torchsnooper.snoop()
    def forward(self, g, gfeat, next_cate):
        '''
        Args:
        ----
            feat (torch.Tensor[bs, embsize]): input feature as anchor
        '''
        # ifeat, vfeat = self.maxpooling_feat(g, gfeat)
        ifeat, vfeat = gfeat['i'], gfeat['v']
        ifeat_u = self.fc_u(ifeat)
        anchor_feat = None
        if self.pooling_type == 'ilast':  # Get the last node as anchor
            last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1,
                                        ntype='i')
            anchor_feat = ifeat[last_nodes]

        elif self.pooling_type == 'imean':
            anchor_feat = F.segment.segment_reduce(g.batch_num_nodes('i'),
                                                   ifeat, 'mean')

        elif self.pooling_type == 'cnext':
            next_nodes = g.filter_nodes(lambda nodes: nodes.data['next'] == 1,
                                        ntype='v')
            anchor_feat = vfeat[next_nodes]

        elif self.pooling_type == 'cmean':
            anchor_feat = F.segment.segment_reduce(
                g.batch_num_nodes('v'), vfeat, 'mean')  # Todo:多个virtual node

        feat_v = self.fc_v(anchor_feat)
        feat_v = dgl.broadcast_nodes(g, feat_v, ntype='i')

        e = self.fc_e(th.sigmoid(ifeat_u + feat_v))
        alpha = F.segment.segment_softmax(g.batch_num_nodes('i'), e / self.tao)
        feat_norm = ifeat * alpha
        rst = F.segment.segment_reduce(g.batch_num_nodes('i'), feat_norm,
                                       'sum')

        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)

        rst = th.cat([rst, anchor_feat], dim=1)
        return rst


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads=1,
                 feat_drop=0.1,
                 attn_drop=0.1,
                 negative_slope=0.2,
                 residual=True,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats,
                                    out_feats * num_heads,
                                    bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats,
                                    out_feats * num_heads,
                                    bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats,
                                out_feats * num_heads,
                                bias=False)
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats, )))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats,
                                        num_heads * out_feats,
                                        bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    pass
                    # raise DGLError('There are 0-in-degree nodes in the graph, '
                    #                'output for those nodes will be invalid. '
                    #                'This is harmful for some applications, '
                    #                'causing silent performance regression. '
                    #                'Adding self-loop on the input graph by '
                    #                'calling `g = dgl.add_self_loop(g)` will resolve '
                    #                'the issue. Setting ``allow_zero_in_degree`` '
                    #                'to be `True` when constructing this module will '
                    #                'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads,
                                                   self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads,
                                                   self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads,
                                                       self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads,
                                                       self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0],
                                                 self._num_heads,
                                                 self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
