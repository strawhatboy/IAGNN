#coding=utf-8
# Main functions:
# dataset collate function, transfer sequence data into graph based on the given functions.
#
# Create by 16525, 2021/6/13 14:57

import torch
import dgl

def gnn_collate_fn(seq_to_graph_fn):
    """
    @param seq_to_graph_fns: a list of funs which transfer a sequence data into a graph
    @return iterate the data with the (graphs,label), where graphs=[fun1(s),fun2(s),...]
    """
    def collate_fn(samples):
        # parse the input data of dataset, samples=[session data]
        uid, browsed_ids, label, cates, seq_len, next_cate = zip(*samples)
        # ipdb.set_trace()
        # data zipped here! tuple(6) -> tuple(4)
        data = zip(browsed_ids, cates, seq_len, next_cate)
        graphs = list(map(seq_to_graph_fn, data))  # run seq_to_graph(data)
        # print(len(graphs))
        inputs = dgl.batch(graphs)
        # print(label)
        label = torch.LongTensor(label)
        next_cate = torch.LongTensor(next_cate)
        # seq_len = torch.LongTensor(np.array(seq_len))
        return inputs, label, next_cate

    return collate_fn
