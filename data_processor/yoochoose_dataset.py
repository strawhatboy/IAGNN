#coding=utf-8
# Main functions:
#   ref:https://github.com/Autumn945/MGNN-SPred/blob/master/dataset.py
# TO DO LIST:
#
# Create by 16525, 2021/3/18 18:35

import numpy as np, sys, math, os
from torch.utils.data import Dataset
import json
import pickle
from typing import List
import copy
import utils
import re
import dgl
from tqdm import tqdm
import time
import ipdb



def gen_seq(data_list,type='aug'):
    '''
    dataset=[[10568, 10568], [1, 1]], [[312, 312], [1, 1]], [[3489, 18083], [1, 1]]
    data=[[10568, 10568], [1, 1]]
    '''
    out_seqs,label = [],[]
    out_behavs,lable_behav = [],[]
    uid = []

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)

    for data in tqdm(data_list[0], desc='aug gen_seq...', leave=False):
        seq, cat = data
        for i in range(1, len(seq) - 1):
            if len(set(cat[:-i+1]))>1:
                uid.append(int(0))
                out_seqs.append(seq[:-i])
                label.append([seq[-i]])
                out_behavs.append(cat[:-i])
                lable_behav.append([cat[-i]])

    sorted_idx = len_argsort(out_seqs)
    final_seqs = []
    for i in sorted_idx:
        final_seqs.append([uid[i], out_seqs[i], label[i], out_behavs[i], lable_behav[i]])
    return final_seqs



def multi_length(data,thresold):      # generate the next behav=buy and next behav=fav
    # data[0]=[out_seqs[i], label[i],out_behavs[i],lable_behav[i]]
    new_data2=[]
    new_data3 = []
    for s in data:
        if len(s[1])<=thresold:
            new_data2.append(s)
        else:
            new_data3.append(s)
    return new_data2,new_data3

def multi_length_detail(data,thresold):      # generate the next behav=buy and next behav=fav
    # data[0]=[out_seqs[i], label[i],out_behavs[i],lable_behav[i]]
    data1,data2,data3=[],[],[]
    for s in data:
        if len(s[1])<=3:
            data1.append(s)
        elif len(s[1])<=6:
            data2.append(s)
        else:
            data3.append(s)
    return data1,data2,data3

def count_times(data):
    iid_counts = {}
    for udata in data.values():
        for s in udata:
            seq = s[0]
            for iid in seq:
                if iid in iid_counts:
                    iid_counts[iid] += 1
                else:
                    iid_counts[iid] = 1
    return iid_counts

def freq_fliter(data,iid_counts,threshold):
    new_data=[]
    for udata in data.values():
        for s in udata:
            seq,cate,times= s
            new_seq,new_behav,new_times=[],[],[]
            for i,id in enumerate(seq):
                if iid_counts[id]>=threshold:
                    new_seq.append(id)
                    new_behav.append(cate[i])
                    new_times.append(times[i])
            new_data.append([new_seq,new_behav,new_times])
    return {0:new_data}

def crossdomain_fliter(data,iid_counts,threshold):
    new_data=[]
    for udata in data.values():
        for s in udata:
            seq, cate, times = s
            new_seq, new_cat = [], []
            if len(list(set(cate))) > 1:
                for i, id in enumerate(seq):
                    if iid_counts[id] >= threshold:
                        new_seq.append(id)
                        new_cat.append(cate[i])
            new_data.append([new_seq, new_cat])
    return {0:new_data}

def get_item_cates(data):       #todo:可以和上面那个函数合并
    item2cates = {}
    max_id=0
    for udata in data.values():
        for s in udata:
            seq = s[0]
            cate=s[-1]
            for i,iid in enumerate(seq):
                if iid not in item2cates:
                    item2cates[iid]=cate[i]
                    max_id=max(max_id,iid)
    item_cates=[]
    max_id += 10
    for i in range(max_id):
        if i in item2cates:
            item_cates.append(item2cates[i])
        else:
            item_cates.append(-1)
    return item_cates,max_id



def next_cate_detail(data: List[List[List]], thresholds: List[int]):
    '''
    split the data by occurence of next_cat in the session

    Args:
        data (List[List[List]]): list of [0, [item], [next_item](1), [category], [next_category](1)]
        thresholds (List[int]): list of thresholds, e.g. [0, 1, 4] means [0, 1-3, 4-∞]
    '''
    # ret = [[], [], []]
    # for s in data:
    #     categories = s[3]
    #     next_cat = s[4][0]
    #     next_cat_count = categories.count(next_cat)
    #     # bisect(*) - 1: get next_cat_count position in thresholds
    #     ret[bisect(thresholds, next_cat_count) - 1].append(s)
    ret = [[], [], [], []]  # -2 and -1 是两种模型：一种是next cate=last cate，另一种是next cate!=last cate
    for s in data:
        categories = s[3]
        next_cat = s[4][0]
        next_cat_count = categories.count(next_cat)
        if next_cat_count == 0:
            ret[0].append(s)
        elif next_cat_count <= 3:
            ret[1].append(s)
        else:
            last_count = 0
            l = len(categories)
            for i in range(l):
                if categories[l - i - 1] == next_cat:
                    last_count += 1
                else:
                    break
            if next_cat_count - last_count >= 1:  # un-continues pattern
                ret[2].append(s)
            else:
                ret[3].append(s)  # chunk pattern

    return tuple(ret)

def load_cd_data(data_path, type='aug', test_length=True, highfreq_only=True):  # todo:改数据通路
    '''
    dataset=[[10568, 10568], [1, 1]], [[312, 312], [1, 1]], [[3489, 18083], [1, 1]]
    data=[[10568, 10568], [1, 1]]
    '''
    print('yoochoose 16!!')
    with open(data_path + '/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    # print(train_data[1])
    if highfreq_only:
        iid_count = count_times(train_data)
        train_data = crossdomain_fliter(train_data, iid_count, 5)

    item_cates,max_vid=get_item_cates(train_data)
    train_data = gen_seq(train_data, type)

    with open(data_path + '/test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    if highfreq_only:
        test_data=crossdomain_fliter(test_data,iid_count,5)

    test_data = gen_seq(test_data, type)

    all_data={"train":train_data, "test":test_data}

    if test_length==True:
        test1, test2, test3, test4 = next_cate_detail(test_data, [0, 1, 4])
        all_data['test1'], all_data['test2'], all_data['test3'], all_data['test4'] = test1, test2, test3, test4
# test1: 10672, test2: 25460, test3: 11712, test4: 11413, total: 59257

    return all_data,max_vid, item_cates


class TBVSessionDataset(Dataset):
    def __init__(self, data, max_length):
        super(TBVSessionDataset, self).__init__()

        self.data = data
        self.max_seq_len=max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = self.data[index]
        uid = np.array([0], dtype=np.int)
        browsed_ids = np.zeros((self.max_seq_len), dtype=np.int)
        cates = browsed_ids.copy()

        seq_len = len(data[1][-self.max_seq_len:])

        browsed_ids[:seq_len] = np.array(data[1][-self.max_seq_len:])
        cates[:seq_len] = np.array(data[3][-self.max_seq_len:])
        next_cate = np.array([data[4][0]],dtype=np.int)

        seq_len = np.array(seq_len, dtype=np.int)
        label = np.array(data[2], dtype=np.int)

        return uid,browsed_ids, label,cates,seq_len,next_cate


class SessionDataset(Dataset):
    def __init__(self, data, args):
        """
        args:
            config(dict):
            data_type(int): 0: train 1: val 2:test
        """
        super(SessionDataset, self).__init__()
        # self.config=config

        self.data = data
        self.max_seq_len = args.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        data format:
        <uid><[v1,v2,v3]> <[label]> #要新增<[vb1,vb2,vb3]> <[lableb]>
        """
        data = self.data[index]
        uid = np.array([0], dtype=np.int)
        browsed_ids = np.zeros((self.max_seq_len), dtype=np.int)
        browsed_behavs = np.zeros((self.max_seq_len), dtype=np.int)

        pos_idx = np.zeros((self.max_seq_len), dtype=np.int)

        # browsed_categ_ids=np.zeros((self.config.history_len),dtype=np.int)

        # browsed_subcateg_ids=np.zeros((self.config.history_len),dtype=np.int)
        # candidate_ids=np.zeros((self.sample_size),dtype=np.int)
        seq_len = len(data[1][-self.max_seq_len:])
        lable_len = len(data[2])
        label_behav = np.zeros((lable_len), dtype=np.int)
        pos_idx[:seq_len] = [seq_len - 1 - _ for _ in range(seq_len)]
        mask = np.array([1 for i in range(seq_len)] +
                        [0 for i in range(self.max_seq_len - seq_len)], dtype=np.int)
        # browsed_mask=torch.ByteTensor([1 for _ in range(x)]+[0 for _ in range(self.history_len-x )])
        browsed_ids[:seq_len] = np.array(data[1][-self.max_seq_len:])
        browsed_behavs[:seq_len] = np.array(data[3][-self.max_seq_len:])
        seq_len = np.array(seq_len, dtype=np.int)

        label = np.array(data[2], dtype=np.int)
        label_behav[:lable_len] = np.array(data[4])

        return uid,browsed_ids, mask, seq_len, label, pos_idx, browsed_behavs, label_behav



