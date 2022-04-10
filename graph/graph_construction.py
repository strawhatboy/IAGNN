#coding=utf-8
# Main functions:
#
# TO DO LIST:
#
# Create by 16525, 2021/6/13 14:55
from typing import List, Tuple
import numpy
import torch
import dgl
import numpy as np
from collections import Counter
import torchsnooper
import ipdb


def label_data(g, items, cates, position, last_nid,seq_cid,next_cid):
    """
    attach the related information to single graph
    :param items: items id for each item node
    :param cates: category inform for related cate node
    :param position: reverse pos inform for relation between item and user
    :param last_nid: last node id => each node is last or not
    :return:
    """
    g.nodes["i"].data['id'] = torch.from_numpy(items)
    # g.nodes['i'].data['cate']=torch.from_numpy(seq_cid)
    g.nodes['v'].data['id'] = torch.from_numpy(cates)

    g.edges["v2i"].data["pos"] = torch.from_numpy(position)

    is_last = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    is_last[last_nid] = 1
    g.nodes['i'].data['last'] = is_last

    is_next = torch.zeros(g.number_of_nodes('v'), dtype=torch.int32)
    is_next[next_cid] = 1
    g.nodes['v'].data['next'] = is_next

    return g

def seq_to_vgraph1(data):
    browsed_ids, cate_ids, seq_len, next_cate = data

    browsed_ids = browsed_ids[:seq_len]
    cate_ids = np.concatenate((cate_ids[:seq_len], next_cate), axis=0)

    positions = np.array([seq_len - 1 - _ for _ in range(seq_len)])

    items, seq_nid = np.unique(browsed_ids, return_inverse=True)

    src = seq_nid[:seq_len - 1]
    dst = seq_nid[1:seq_len]
    # construction the bi-directed graph
    src1 = np.concatenate((src, dst), axis=0)
    dst1 = np.concatenate((dst, src), axis=0)

    cates, seq_cid = np.unique(cate_ids, return_inverse=True)
    next_cid = seq_cid[-1]
    seq_cid = seq_cid[:-1]

    g = dgl.heterograph({("i", "i2i", "i"): (src1, dst1),
                         ("v", "v2i", "i"): (seq_cid, seq_nid),
                         ("i", "i2v", "v"): (seq_nid, seq_cid)},
                        num_nodes_dict={'i': len(items), 'v': len(cates)})

    g = label_data(g, items, cates, positions, seq_nid[-1],seq_cid, next_cid)
    return g




def label_data2(g, items, cates, position, last_nid, next_cid):     #todo：附着不同的relation for i2v
    """
    attach the related information to single graph
    :param items: items id for each item node
    :param cates: category inform for related cate node
    :param position: reverse pos inform for relation between item and user
    :param last_nid: last node id => each node is last or not
    :return:
    """
    g.nodes["i"].data['id'] = torch.from_numpy(items)
    g.nodes['v'].data['id'] = torch.from_numpy(cates)

    g.edges["v2i"].data["pos"] = torch.from_numpy(position)

    is_last = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    is_last[last_nid] = 1
    g.nodes['i'].data['last'] = is_last

    is_next = torch.zeros(g.number_of_nodes('v'), dtype=torch.int32)
    is_next[next_cid] = 1
    g.nodes['v'].data['next'] = is_next

    return g


def seq_to_vgraph2(data):
    browsed_ids, cate_ids, seq_len, next_cate = data

    browsed_ids = browsed_ids[:seq_len]
    cate_ids = np.concatenate((cate_ids[:seq_len], next_cate), axis=0)

    positions = np.array([seq_len - 1 - _ for _ in range(seq_len)])

    items, seq_nid = np.unique(browsed_ids, return_inverse=True)

    src = seq_nid[:seq_len - 1]
    dst = seq_nid[1:seq_len]
    # construction the bi-directed graph
    src1 = np.concatenate((src, dst), axis=0)
    dst1 = np.concatenate((dst, src), axis=0)

    cates, seq_cid = np.unique(cate_ids, return_inverse=True)
    next_cid = seq_cid[-1]
    seq_cid = seq_cid[:-1]

    cats, cat_nid = np.unique(cate_ids, return_inverse=True)
    cat_nid = cat_nid[:-1]
    cat_nid_u = np.unique(cat_nid)

    # map_c_items = {}
    last_id = -1
    v2i_full_conn = [np.array([], dtype='int64'), np.array([], dtype='int64')]
    for c in cat_nid_u:
        mask = cat_nid == c
        item_nid_in_c = seq_nid[mask]

        if len(item_nid_in_c) > 0:
            
            if c == next_cid:
                last_id = item_nid_in_c[-1]
        
        item_nid_not_in_c = seq_nid[cat_nid != c]
        v2i_full_conn[0] = np.concatenate((v2i_full_conn[0], np.array([c] * len(item_nid_not_in_c), dtype='int64')))
        v2i_full_conn[1] = np.concatenate((v2i_full_conn[1], item_nid_not_in_c))
            # map_c_items[c] = item_nid_in_c
    
    # no last item of next category, use the last item of the whole sequence.
    if last_id == -1:
        last_id = seq_nid[-1]

    g = dgl.heterograph({("i", "i2i", "i"): (src1, dst1),
                         ("v", "v2i", "i"): (seq_cid, seq_nid),
                         ("i", "i2v", "v"): (seq_nid, seq_cid),
                         ("v", 'v2if', 'i'): tuple(v2i_full_conn),
                         },
                        num_nodes_dict={'i': len(items), 'v': len(cates)})
    # g = dgl.heterograph({("i", "i2i", "i"): (src1, dst1),       #todo:增加v2i的full connection
    #                      ("v", "v2i", "i"):[],
    #                      ("i", "i2v", "v"):[]},
    #                     num_nodes_dict={'i': len(items), 'v': len(cates)})

    g = label_data2(g, items, cates, positions, last_id, next_cid)
    return g


def label_last(g, last_nid, next_cat):
    is_last = torch.zeros(g.number_of_nodes(), dtype=torch.int32)
    is_last[last_nid] = 1
    next = torch.zeros(g.number_of_nodes(), dtype=torch.int32)
    next[last_nid] = next_cat
    g.ndata['last'] = is_last
    g.ndata['next'] = next
    return g

def seq_to_eop_multigraph(seq):

    brows_id, cat_id, seq_len, next_cat = seq
    next_cat = next_cat[0]
    seq=brows_id[:seq_len]
    items = np.unique(seq)
    # print(items)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
    else:
        src = torch.LongTensor([])
        dst = torch.LongTensor([])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = torch.from_numpy(items)
    label_last(g, iid2nid[seq[-1]], next_cat)
    return g


def seq_to_shortcut_graph(seq):
    brows_id, cat_id, seq_len, next_cat = seq
    seq=brows_id[:seq_len]

    items, seq_nid = np.unique(seq, return_inverse=True)
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    edges = counter.keys()
    src, dst = zip(*edges)

    g = dgl.graph((src, dst), num_nodes=len(items))
    return g


def label_data_CDS(g: dgl.DGLHeteroGraph, items, map_c_items, positions, next_cate, last_id, seq_last_id):
    '''
    Args:
    ----
        items (np.array): item ids
        map_c_items (Dict[c_nid, item_nid]): category nid related item nids
        positions (np.array): positions in each sequence for each category
        next_cate (int): next category id
        last_id (int): last item in 'next category' nid
    '''
    g.nodes["i"].data['id'] = torch.from_numpy(items)
    g.nodes["v"].data['id'] = torch.IntTensor([next_cate])

    g.edges['v2i'].data['pos'] = torch.from_numpy(positions)

    items_cat_label = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    for c, c_items in map_c_items.items():
        items_cat_label[c_items] = c
    g.nodes['i'].data['cate'] = items_cat_label

    last_id_label = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    last_id_label[last_id] = 1
    g.nodes['i'].data['last'] = last_id_label

    seq_last_id_label = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    seq_last_id_label[seq_last_id] = 1
    g.nodes['i'].data['seq_last'] = seq_last_id_label
    
    g.nodes['i'].data['next'] = torch.ones(g.number_of_nodes('i'), dtype=torch.int32) * next_cate
    g.nodes['v'].data['next'] = torch.LongTensor([1])
    # only one v node, won't label it
    return g

def seq_to_CDS_graph(data: Tuple[List, List, int, int]) -> dgl.DGLHeteroGraph:
    '''
    For DAGCN_Origin, v1, v2, v3
    `CDS` (Cross-Domain Sequence) introduced in `DA-GCN` (Domain-aware Attentive GCN).

    Args:
        `data (Tuple[List, List, int, int])`: ([item](10), [catetory], length, next_category)
        was generated by collate.py/vgnn_collate_fn
        
    Returns:
        `dgl.DGLHeteroGraph`
    '''
    item_ids, cate_ids, seq_len, next_cate = data
    item_ids = item_ids[:seq_len]
    cate_ids = np.concatenate((cate_ids[:seq_len], next_cate), axis=0)
    e_i2i = [np.array([], dtype='int64'), np.array([], dtype='int64')]
    e_mp = [np.array([], dtype='int64'), np.array([], dtype='int64')]

    # positions = np.array([], dtype='int64')

    # use original position in the sequence
    positions = np.array([seq_len - 1 - _ for _ in range(seq_len)], dtype='float')
    cats, cat_nid = np.unique(cate_ids, return_inverse=True)
    items, item_nid = np.unique(item_ids, return_inverse=True)

    next_cid = cat_nid[-1]
    cat_nid = cat_nid[:-1]

    # positions = np.array([seq_len - 1 - _ for _ in range(seq_len)])
    # cat_nid += 1
    cat_nid_u = np.unique(cat_nid)

    map_c_items = {}
    last_id = -1
    for c in cat_nid_u:
        mask = cat_nid == c
        item_nid_in_c = item_nid[mask]

        if len(item_nid_in_c) > 0:
            # single direction seq i2i edges in the category c
            # and self connections
            e_i2i[0] = np.concatenate((e_i2i[0], item_nid_in_c[:-1], item_nid_in_c), axis=0)
            e_i2i[1] = np.concatenate((e_i2i[1], item_nid_in_c[1:], item_nid_in_c), axis=0)

            # in this category c, every item linked to the last item for max pooling
            e_mp[0] = np.concatenate((e_mp[0], item_nid_in_c), axis=0)
            e_mp[1] = np.concatenate((e_mp[1], np.array([item_nid_in_c[-1]] * len(item_nid_in_c), dtype='int64')))
            # positions = np.concatenate((positions, [len(item_nid_in_c) - 1 - _ for _ in range(len(item_nid_in_c))]))
            
            if c == next_cid:
                last_id = item_nid_in_c[-1]
            map_c_items[c] = item_nid_in_c
    
    # no last item of next category, use the last item of the whole sequence.
    if last_id == -1:
        last_id = item_nid[-1]

    seq_last_id = item_nid[-1]

    # all item connected to v node, vice versa
    vnodes = np.zeros_like(item_nid)

    # construct the graph
    g = dgl.heterograph({("i", "i2i", "i"): tuple(e_i2i),
                         ("v", "v2i", "i"): (vnodes, item_nid),
                         ("i", "i2v", "v"): (item_nid, vnodes),
                         ("v", "v2v", "v"): ([0], [0]), # vnode self connection
                         ("i", "e", "i"): tuple(e_i2i),
                         ("v", "e", "i"): (vnodes, item_nid),
                         ("i", "e", "v"): (item_nid, vnodes),
                         ("v", "e", "v"): ([0], [0]), # vnode self connection
                         ('i', 'mp', 'i'): tuple(e_mp), # maxpooling connections
                         },
                        num_nodes_dict={'i': len(items), 'v': 1})

    g = label_data_CDS(g, items, map_c_items, positions, next_cate, last_id, seq_last_id)
    
    return g

def label_data_Star(g: dgl.DGLHeteroGraph, items, map_c_items, positions, next_cid, last_id, seq_last_id):
    '''
    Args:
    ----
        items (np.array): item ids
        map_c_items (Dict[c_nid, item_nid]): category nid related item nids
        positions (np.array): positions in each sequence for each category
        next_cid (int): next category id
        last_id (int): last item in 'next category' nid
    '''
    g.nodes["i"].data['id'] = torch.from_numpy(items)
    g.nodes["v"].data['id'] = torch.IntTensor([next_cid])

    g.edges['v2i'].data['pos'] = torch.from_numpy(positions)

    items_cat_label = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    for c, c_items in map_c_items.items():
        items_cat_label[c_items] = c
    g.nodes['i'].data['cate'] = items_cat_label

    last_id_label = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    last_id_label[last_id] = 1
    g.nodes['i'].data['last'] = last_id_label

    seq_last_id_label = torch.zeros(g.number_of_nodes('i'), dtype=torch.int32)
    seq_last_id_label[seq_last_id] = 1
    g.nodes['i'].data['seq_last'] = seq_last_id_label
    
    g.nodes['i'].data['next'] = torch.ones(g.number_of_nodes('i'), dtype=torch.int32) * next_cid
    g.nodes['v'].data['next'] = torch.LongTensor([1])
    # only one v node, won't label it
    return g

def seq_to_Star_graph(data: Tuple[List, List, int, int]) -> dgl.DGLHeteroGraph:
    '''
    For star graph
    `CDS` (Cross-Domain Sequence) introduced in `DA-GCN` (Domain-aware Attentive GCN).

    Args:
        `data (Tuple[List, List, int, int])`: ([item](10), [catetory], length, next_category)
        was generated by collate.py/vgnn_collate_fn
        
    Returns:
        `dgl.DGLHeteroGraph`
    '''
    item_ids, cate_ids, seq_len, next_cate = data
    item_ids = item_ids[:seq_len]
    cate_ids = np.concatenate((cate_ids[:seq_len], next_cate), axis=0)
    e_i2i = [np.array([], dtype='int64'), np.array([], dtype='int64')]
    e_mp = [np.array([], dtype='int64'), np.array([], dtype='int64')]

    # positions = np.array([], dtype='int64')

    # use original position in the sequence
    positions = np.array([seq_len - 1 - _ for _ in range(seq_len)], dtype='float')
    cats, cat_nid = np.unique(cate_ids, return_inverse=True)
    items, item_nid = np.unique(item_ids, return_inverse=True)

    next_cid = cat_nid[-1]
    cat_nid = cat_nid[:-1]

    # positions = np.array([seq_len - 1 - _ for _ in range(seq_len)])
    # cat_nid += 1
    cat_nid_u = np.unique(cat_nid)

    map_c_items = {}
    last_id = -1
    # for c in cat_nid_u:
    #     mask = cat_nid == c
    #     item_nid_in_c = item_nid[mask]

    #     if len(item_nid_in_c) > 0:
    #         e_i2i[0] = np.concatenate((e_i2i[0], item_nid_in_c[:-1], item_nid_in_c), axis=0)
    #         e_i2i[1] = np.concatenate((e_i2i[1], item_nid_in_c[1:], item_nid_in_c), axis=0)
    #         # in this category c, every item linked to the last item for max pooling
    #         e_mp[0] = np.concatenate((e_mp[0], item_nid_in_c), axis=0)
    #         e_mp[1] = np.concatenate((e_mp[1], np.array([item_nid_in_c[-1]] * len(item_nid_in_c), dtype='int64')))
    #         # positions = np.concatenate((positions, [len(item_nid_in_c) - 1 - _ for _ in range(len(item_nid_in_c))]))
            
    #         if c == next_cid:
    #             last_id = item_nid_in_c[-1]
    #         map_c_items[c] = item_nid_in_c
    
    # no last item of next category, use the last item of the whole sequence.
    if last_id == -1:
        last_id = item_nid[-1]

    seq_last_id = item_nid[-1]

    # all item connected to v node, vice versa
    vnodes = np.zeros_like(item_nid)

    e_i2i[0] = np.concatenate((e_i2i[0], item_nid[:-1]), axis=0)
    e_i2i[1] = np.concatenate((e_i2i[1], item_nid[1:]), axis=0)

    # construct the graph
    g = dgl.heterograph({("i", "i2i", "i"): tuple(e_i2i),
                         ("v", "v2i", "i"): (vnodes, item_nid),
                         ("i", "i2v", "v"): (item_nid, vnodes),
                         ("v", "v2v", "v"): ([0], [0]), # vnode self connection
                        #  ("i", "e", "i"): tuple(e_i2i),
                        #  ("v", "e", "i"): (vnodes, item_nid),
                        #  ("i", "e", "v"): (item_nid, vnodes),
                        #  ("v", "e", "v"): ([0], [0]), # vnode self connection
                        #  ('i', 'mp', 'i'): tuple(e_mp), # maxpooling connections
                         },
                        num_nodes_dict={'i': len(items), 'v': 1})

    g = label_data_Star(g, items, map_c_items, positions, next_cid, last_id, seq_last_id)
    
    return g


def label_data_SSL(g: dgl.DGLHeteroGraph, items, cates, map_c_items, positions, next_cate, last_id, seq_last_id, last_cid, seq_last_cid):
    '''
    Args:
    ----
        items (np.array): item ids
        map_c_items (Dict[c_nid, item_nid]): category nid related item nids
        positions (np.array): positions in each sequence for each category
        next_cate (int): next category id
        last_id (int): last item in 'next category' nid
    '''
    g.nodes["i"].data['id'] = torch.from_numpy(items)
    g.nodes["v"].data['id'] = torch.IntTensor([next_cate])
    g.nodes['c'].data['id'] = torch.from_numpy(cates)

    g.edges['v2i'].data['pos'] = torch.from_numpy(positions)

    i_count = len(items)
    c_count = len(cates)

    items_cat_label = torch.zeros(i_count, dtype=torch.int32)
    for c, c_items in map_c_items.items():
        items_cat_label[c_items] = c
    g.nodes['i'].data['cate'] = items_cat_label

    last_id_label = torch.zeros(i_count, dtype=torch.int32)
    last_id_label[last_id] = 1
    g.nodes['i'].data['last'] = last_id_label
    
    last_cid_label = torch.zeros(c_count, dtype=torch.int32)
    last_cid_label[last_cid] = 1
    g.nodes['c'].data['clast'] = last_cid_label

    seq_last_id_label = torch.zeros(i_count, dtype=torch.int32)
    seq_last_id_label[seq_last_id] = 1
    g.nodes['i'].data['seq_last'] = seq_last_id_label

    seq_last_cid_label = torch.zeros(c_count, dtype=torch.int32)
    seq_last_cid_label[seq_last_cid] = 1
    g.nodes['c'].data['seq_clast'] = seq_last_cid_label
    
    # g.nodes['i'].data['next'] = torch.ones(g.number_of_nodes('i'), dtype=torch.int32) * next_cid
    g.nodes['v'].data['next'] = torch.LongTensor([1])
    # only one v node, won't label it
    return g


def seq_to_SSL_graph(data: Tuple[List, List, int, int]) -> dgl.DGLHeteroGraph:
    '''
    For DAGCN_v4
    `SSL` (Self-Supervised Learning Graph Co-Training) introduced in SSGC4SR(Self-Supervised Graph Co-Training for Session-based Recommendation).

    Args:
        `data (Tuple[List, List, int, int])`: ([item](10), [catetory], length, next_category)
        was generated by collate.py/vgnn_collate_fn
        
    Returns:
        `dgl.DGLHeteroGraph`
    '''
    item_ids, cate_ids, seq_len, next_cate = data
    item_ids = item_ids[:seq_len]
    cate_ids = np.concatenate((cate_ids[:seq_len], next_cate), axis=0)
    e_i2i = [np.array([], dtype='int'), np.array([], dtype='int')]
    e_c2c = [0, 0]
    # e_mp = [np.array([], dtype='int'), np.array([], dtype='int')]

    # positions = np.array([], dtype='int64')

    # use original position in the sequence
    positions = np.array([seq_len - 1 - _ for _ in range(seq_len)], dtype='float')
    cats, cat_nid = np.unique(cate_ids, return_inverse=True)
    items, item_nid = np.unique(item_ids, return_inverse=True)

    next_cid = cat_nid[-1]
    cat_nid = cat_nid[:-1]

    # positions = np.array([seq_len - 1 - _ for _ in range(seq_len)])
    # cat_nid += 1
    cat_nid_u = np.unique(cat_nid)

    map_c_items = {}
    last_id = -1
    for c in cat_nid_u:
        mask = cat_nid == c
        item_nid_in_c = item_nid[mask]

        if len(item_nid_in_c) > 0:
            # single direction seq i2i edges in the category c
            # and self connections
            e_i2i[0] = np.concatenate((e_i2i[0], item_nid_in_c[:-1], item_nid_in_c), axis=0)
            e_i2i[1] = np.concatenate((e_i2i[1], item_nid_in_c[1:], item_nid_in_c), axis=0)

            # in this category c, every item linked to the last item for max pooling
            # e_mp[0] = np.concatenate((e_mp[0], item_nid_in_c), axis=0)
            # e_mp[1] = np.concatenate((e_mp[1], np.array([item_nid_in_c[-1]] * len(item_nid_in_c), dtype='int')))
            # positions = np.concatenate((positions, [len(item_nid_in_c) - 1 - _ for _ in range(len(item_nid_in_c))]))
            
            if c == next_cid:
                last_id = item_nid_in_c[-1]
                last_cid = c
            map_c_items[c] = item_nid_in_c
    
    # no last item of next category, use the last item of the whole sequence.
    if last_id == -1:
        last_id = item_nid[-1]
        last_cid = cat_nid[-1]

    seq_last_id = item_nid[-1]
    seq_last_cid = cat_nid[-1]

    # all item connected to v node, vice versa
    vnodes = np.zeros_like(item_nid)

    # category sequence and self-connection
    e_c2c[0] = np.concatenate((cat_nid[:-1], cat_nid))
    e_c2c[1] = np.concatenate((cat_nid[1:], cat_nid))

    # also add the origin seq links
    # e_i2i[0] = np.concatenate((e_i2i[0], item_nid[:-1]), axis=0)
    # e_i2i[1] = np.concatenate((e_i2i[1], item_nid[1:]), axis=0)

    # c2c_edges_count = len(e_c2c[0])
    # for c in cat_nid_u:
    #     c_in_edges_count = np.sum(cat_nid[1:] == c) + 1


    # construct the graph
    g = dgl.heterograph({("i", "i2i", "i"): tuple(e_i2i),
                         ("v", "v2i", "i"): (vnodes, item_nid),
                         ("i", "i2v", "v"): (item_nid, vnodes),
                         ("v", "v2v", "v"): ([0], [0]), # vnode self connection
                        #  ("i", "e", "i"): tuple(e_i2i),
                        #  ("v", "e", "i"): (vnodes, item_nid),
                        #  ("i", "e", "v"): (item_nid, vnodes),
                        #  ("v", "e", "v"): ([0], [0]), # vnode self connection
                        #  ('i', 'mp', 'i'): tuple(e_mp), # maxpooling connections
                         ('i', 'i2c', 'c'): (item_nid, cat_nid),
                         ('c', 'c2i', 'i'): (cat_nid, item_nid),
                         ('c', 'c2c', 'c'): tuple(e_c2c),
                        #  ('i', 'e2', 'c'): (item_nid, cat_nid),
                        #  ('c', 'e', 'i'): (cat_nid, item_nid),
                        #  ('c', 'e2', 'c'): tuple(e_c2c)
                         },
                        num_nodes_dict={'i': len(items), 'v': 1, 'c': len(cats)})

    g = label_data_SSL(g, items, cats, map_c_items, positions, next_cate, last_id, seq_last_id, last_cid, seq_last_cid)
    
    return g

def seq_to_SSL_graph_ppos(data: Tuple[List, List, int, int]) -> dgl.DGLHeteroGraph:
    '''
    For DAGCN_v4
    `SSL` (Self-Supervised Learning Graph Co-Training) introduced in SSGC4SR(Self-Supervised Graph Co-Training for Session-based Recommendation).

    Args:
        `data (Tuple[List, List, int, int])`: ([item](10), [catetory], length, next_category)
        was generated by collate.py/vgnn_collate_fn
        
    Returns:
        `dgl.DGLHeteroGraph`
    '''
    item_ids, cate_ids, seq_len, next_cate = data
    item_ids = item_ids[:seq_len]
    cate_ids = np.concatenate((cate_ids[:seq_len], next_cate), axis=0)
    e_i2i = [np.array([], dtype='int'), np.array([], dtype='int')]
    e_c2c = [0, 0]
    # e_mp = [np.array([], dtype='int'), np.array([], dtype='int')]

    # positions = np.array([], dtype='int64')

    # use original position in the sequence
    positions = np.array([_ for _ in range(seq_len)], dtype='float')
    cats, cat_nid = np.unique(cate_ids, return_inverse=True)
    items, item_nid = np.unique(item_ids, return_inverse=True)

    next_cid = cat_nid[-1]
    cat_nid = cat_nid[:-1]

    # positions = np.array([seq_len - 1 - _ for _ in range(seq_len)])
    # cat_nid += 1
    cat_nid_u = np.unique(cat_nid)

    map_c_items = {}
    last_id = -1
    for c in cat_nid_u:
        mask = cat_nid == c
        item_nid_in_c = item_nid[mask]

        if len(item_nid_in_c) > 0:
            # single direction seq i2i edges in the category c
            # and self connections
            e_i2i[0] = np.concatenate((e_i2i[0], item_nid_in_c[:-1], item_nid_in_c), axis=0)
            e_i2i[1] = np.concatenate((e_i2i[1], item_nid_in_c[1:], item_nid_in_c), axis=0)

            # in this category c, every item linked to the last item for max pooling
            # e_mp[0] = np.concatenate((e_mp[0], item_nid_in_c), axis=0)
            # e_mp[1] = np.concatenate((e_mp[1], np.array([item_nid_in_c[-1]] * len(item_nid_in_c), dtype='int')))
            # positions = np.concatenate((positions, [len(item_nid_in_c) - 1 - _ for _ in range(len(item_nid_in_c))]))
            
            if c == next_cid:
                last_id = item_nid_in_c[-1]
                last_cid = c
            map_c_items[c] = item_nid_in_c
    
    # no last item of next category, use the last item of the whole sequence.
    if last_id == -1:
        last_id = item_nid[-1]
        last_cid = cat_nid[-1]

    seq_last_id = item_nid[-1]
    seq_last_cid = cat_nid[-1]

    # all item connected to v node, vice versa
    vnodes = np.zeros_like(item_nid)

    # category sequence and self-connection
    e_c2c[0] = np.concatenate((cat_nid[:-1], cat_nid))
    e_c2c[1] = np.concatenate((cat_nid[1:], cat_nid))

    # also add the origin seq links
    # e_i2i[0] = np.concatenate((e_i2i[0], item_nid[:-1]), axis=0)
    # e_i2i[1] = np.concatenate((e_i2i[1], item_nid[1:]), axis=0)

    # c2c_edges_count = len(e_c2c[0])
    # for c in cat_nid_u:
    #     c_in_edges_count = np.sum(cat_nid[1:] == c) + 1


    # construct the graph
    g = dgl.heterograph({("i", "i2i", "i"): tuple(e_i2i),
                         ("v", "v2i", "i"): (vnodes, item_nid),
                         ("i", "i2v", "v"): (item_nid, vnodes),
                         ("v", "v2v", "v"): ([0], [0]), # vnode self connection
                        #  ("i", "e", "i"): tuple(e_i2i),
                        #  ("v", "e", "i"): (vnodes, item_nid),
                        #  ("i", "e", "v"): (item_nid, vnodes),
                        #  ("v", "e", "v"): ([0], [0]), # vnode self connection
                        #  ('i', 'mp', 'i'): tuple(e_mp), # maxpooling connections
                         ('i', 'i2c', 'c'): (item_nid, cat_nid),
                         ('c', 'c2i', 'i'): (cat_nid, item_nid),
                         ('c', 'c2c', 'c'): tuple(e_c2c),
                        #  ('i', 'e2', 'c'): (item_nid, cat_nid),
                        #  ('c', 'e', 'i'): (cat_nid, item_nid),
                        #  ('c', 'e2', 'c'): tuple(e_c2c)
                         },
                        num_nodes_dict={'i': len(items), 'v': 1, 'c': len(cats)})

    g = label_data_SSL(g, items, cats, map_c_items, positions, next_cate, last_id, seq_last_id, last_cid, seq_last_cid)
    
    return g

    
def label_data_SSL_without_V(g: dgl.DGLHeteroGraph, items, cates, map_c_items, positions, next_cate, next_cid, last_id, seq_last_id, last_cid, seq_last_cid):
    '''
    Args:
    ----
        items (np.array): item ids
        map_c_items (Dict[c_nid, item_nid]): category nid related item nids
        positions (np.array): positions in each sequence for each category
        next_cate (int): next category id
        next_cid (int): next category node id in the graph
        last_id (int): last item in 'next category' nid
    '''
    g.nodes["i"].data['id'] = torch.from_numpy(items)
    # g.nodes["v"].data['id'] = torch.IntTensor([next_cate])
    g.nodes['c'].data['id'] = torch.from_numpy(cates)
    # g.nodes["i"].data['pos'] = torch.from_numpy(positions)
    g.edges['v2i'].data['pos'] = torch.from_numpy(positions)

    i_count = len(items)
    c_count = len(cates)

    items_cat_label = torch.zeros(i_count, dtype=torch.int32)
    for c, c_items in map_c_items.items():
        items_cat_label[c_items] = c
    g.nodes['i'].data['cate'] = items_cat_label

    last_id_label = torch.zeros(i_count, dtype=torch.int32)
    last_id_label[last_id] = 1
    g.nodes['i'].data['last'] = last_id_label
    
    last_cid_label = torch.zeros(c_count, dtype=torch.int32)
    last_cid_label[last_cid] = 1
    g.nodes['c'].data['clast'] = last_cid_label

    seq_last_id_label = torch.zeros(i_count, dtype=torch.int32)
    seq_last_id_label[seq_last_id] = 1
    g.nodes['i'].data['seq_last'] = seq_last_id_label

    seq_last_cid_label = torch.zeros(c_count, dtype=torch.int32)
    seq_last_cid_label[seq_last_cid] = 1
    g.nodes['c'].data['seq_clast'] = seq_last_cid_label

    next_cid_label = torch.zeros(c_count, dtype=torch.int32)
    next_cid_label[next_cid] = 1
    g.nodes['c'].data['next_cate'] = next_cid_label
    
    # g.nodes['i'].data['next'] = torch.ones(g.number_of_nodes('i'), dtype=torch.int32) * next_cid
    # g.nodes['v'].data['next'] = torch.LongTensor([1])
    # only one v node, won't label it
    return g


def seq_to_SSL_graph_without_V(data: Tuple[List, List, int, int]) -> dgl.DGLHeteroGraph:
    '''
    For DAGCN_v4
    `SSL` (Self-Supervised Learning Graph Co-Training) introduced in SSGC4SR(Self-Supervised Graph Co-Training for Session-based Recommendation).

    Args:
        `data (Tuple[List, List, int, int])`: ([item](10), [catetory], length, next_category)
        was generated by collate.py/vgnn_collate_fn
        
    Returns:
        `dgl.DGLHeteroGraph`
    '''
    item_ids, cate_ids, seq_len, next_cate = data
    item_ids = item_ids[:seq_len]
    cate_ids = np.concatenate((cate_ids[:seq_len], next_cate), axis=0)
    e_i2i = [np.array([], dtype='int'), np.array([], dtype='int')]
    e_c2c = [0, 0]
    # e_mp = [np.array([], dtype='int'), np.array([], dtype='int')]

    # positions = np.array([], dtype='int64')

    # use original position in the sequence
    positions = np.array([seq_len - 1 - _ for _ in range(seq_len)], dtype='float')
    cats, cat_nid = np.unique(cate_ids, return_inverse=True)
    items, item_nid = np.unique(item_ids, return_inverse=True)

    next_cid = cat_nid[-1]
    cat_nid = cat_nid[:-1]

    # positions = np.array([seq_len - 1 - _ for _ in range(seq_len)])
    # cat_nid += 1
    cat_nid_u = np.unique(cat_nid)

    map_c_items = {}
    last_id = -1
    for c in cat_nid_u:
        mask = cat_nid == c
        item_nid_in_c = item_nid[mask]

        if len(item_nid_in_c) > 0:
            # single direction seq i2i edges in the category c
            # and self connections
            e_i2i[0] = np.concatenate((e_i2i[0], item_nid_in_c[:-1], item_nid_in_c), axis=0)
            e_i2i[1] = np.concatenate((e_i2i[1], item_nid_in_c[1:], item_nid_in_c), axis=0)

            # in this category c, every item linked to the last item for max pooling
            # e_mp[0] = np.concatenate((e_mp[0], item_nid_in_c), axis=0)
            # e_mp[1] = np.concatenate((e_mp[1], np.array([item_nid_in_c[-1]] * len(item_nid_in_c), dtype='int')))
            # positions = np.concatenate((positions, [len(item_nid_in_c) - 1 - _ for _ in range(len(item_nid_in_c))]))
            
            if c == next_cid:
                last_id = item_nid_in_c[-1]
                last_cid = c
            map_c_items[c] = item_nid_in_c
    
    # no last item of next category, use the last item of the whole sequence.
    if last_id == -1:
        last_id = item_nid[-1]
        last_cid = cat_nid[-1]

    seq_last_id = item_nid[-1]
    seq_last_cid = cat_nid[-1]

    # all item connected to v node, vice versa
    vnodes = np.zeros_like(item_nid)

    # category sequence and self-connection
    e_c2c[0] = np.concatenate((cat_nid[:-1], cat_nid))
    e_c2c[1] = np.concatenate((cat_nid[1:], cat_nid))

    # also add the origin seq links
    e_i2i[0] = np.concatenate((e_i2i[0], item_nid[:-1]), axis=0)
    e_i2i[1] = np.concatenate((e_i2i[1], item_nid[1:]), axis=0)

    # c2c_edges_count = len(e_c2c[0])
    # for c in cat_nid_u:
    #     c_in_edges_count = np.sum(cat_nid[1:] == c) + 1

    # construct the graph
    g = dgl.heterograph({("i", "i2i", "i"): tuple(e_i2i),
                         ("v", "v2i", "i"): (vnodes, item_nid),
                        #  ("i", "i2v", "v"): (item_nid, vnodes),
                        #  ("v", "v2v", "v"): ([0], [0]), # vnode self connection
                         ("i", "e", "i"): tuple(e_i2i),
                        #  ("v", "e", "i"): (vnodes, item_nid),
                        #  ("i", "e", "v"): (item_nid, vnodes),
                        #  ("v", "e", "v"): ([0], [0]), # vnode self connection
                        #  ('i', 'mp', 'i'): tuple(e_mp), # maxpooling connections
                         ('i', 'i2c', 'c'): (item_nid, cat_nid),
                         ('c', 'c2i', 'i'): (cat_nid, item_nid),
                         ('c', 'c2c', 'c'): tuple(e_c2c),
                        #  ('i', 'i2v', 'c'): (item_nid, vnodes),
                        #  ('c', 'v2i', 'i'): (vnodes, item_nid),
                        #  ('c', 'v2v', 'c'): ([next_cid], [next_cid])
                        #  ('i', 'e2', 'c'): (item_nid, cat_nid),
                        #  ('c', 'e', 'i'): (cat_nid, item_nid),
                        #  ('c', 'e2', 'c'): tuple(e_c2c)
                         },
                        num_nodes_dict={'i': len(items), 'c': len(cats), 'v': 1})

    g = label_data_SSL_without_V(g, items, cats, map_c_items, positions, next_cate, next_cid, last_id, seq_last_id, last_cid, seq_last_cid)
    
    return g
