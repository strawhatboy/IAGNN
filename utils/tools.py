"""Toolkit for data processing and model.

"""

import time
import functools
from datetime import timedelta
import torch
import dgl
import os
from torch.utils.data import DataLoader
 
def log_exec_time(func):
    """wrapper for log the execution time of function
    
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print('Current Func : {}...'.format(func.__name__))
        start=time.perf_counter()
        res=func(*args,**kwargs)
        end=time.perf_counter()
        print('Func {} took {:.2f}s'.format(func.__name__,(end-start)))
        return res
    return wrapper

def get_time_dif(start_time):
    """calculate the time cost from the start point.
    
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def path_check(pathes):
    for path in pathes:
        if not os.path.exists(path):
            os.makedirs(path)


def set_seed(seed=422):
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # build_dataset(config,_type=0)
    torch.backends.cudnn.deterministic = True


def data_describe(dataset,datas,describes=None):
    print('current dataset:', dataset)
    for key in datas:
        print('The data size of '+key+" :", len(datas[key]))



def datasets(all_data,sessiondataset,max_length,max_vid):
    datas={}
    for key in all_data:
        dataset=sessiondataset(all_data[key],max_length)
        # else:
        #     dataset=sessiondataset(all_data[key],5)
        datas[key]=dataset
    num_class = {"item": max_vid, "pos": 20, 'cate': 1300}
    return datas,num_class


def dataloaders(datas,batch=512,collate=None):
    """
    @param test_datas: dict of all data
    @return: dcit of data iter
    """
    iters={}
    for key in datas:
        iter=DataLoader(dataset=datas[key],
                       batch_size=(batch if key=='train' else batch*2),
                       num_workers=8,
                       drop_last=False,
                       shuffle=(key=='train'),
                       pin_memory=False,collate_fn=collate)
        iters[key]=iter
    return iters


