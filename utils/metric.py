#coding=utf-8
# Main functions:
#
# TO DO LIST:
#
# Create by 16525, 2021/2/28 20:44

import numpy as np
import ipdb
from sklearn.metrics import accuracy_score, classification_report

def metrics(res, labels):

    res = np.concatenate(res)
    # ipdb.set_trace()

    acc_ar = (res == labels.reshape((-1,1))).astype(int)   # [BS, K]
    acc = acc_ar.sum(-1)   #[BS]

    rank = np.argmax(acc_ar, -1) + 1
    mrr = (acc / rank).mean()
    ndcg = (acc / np.log2(rank + 1)).mean()
    return acc.mean(), mrr, ndcg



def metrics2(res, labels):

    res = np.concatenate(res)
    print(classification_report(labels, res))
    return
