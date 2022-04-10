#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as TFn
import numpy as np
from tqdm import tqdm
import dgl
import time
import argparse
from data_processor.data_statistics import data_statistics
from utils.tools import get_time_dif, set_seed, data_describe, dataloaders, datasets, path_check
from utils.logger import Logger
from utils.optim import fix_weight_decay
from utils.metric import metrics
from utils import RunRecordManager
import data_processor.yoochoose_dataset as yoochoose
import data_processor.jdata_dataset as jdata
from graph.graph_construction import *
from graph.collate import gnn_collate_fn
from IAGNN import IAGNN
import pretty_errors

MODEL_NAME = 'DAGCN'


#@torchsnooper.snoop()
def train(args, model, optimizer, scheduler, device, iters, args_filter,
          item_cates):
    model_name = args.model_name
    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss, best_acc = float('inf'), 0
    STEP_SIZE = 200

    last_improve = 0  # 记录上次验证集loss下降的batch
    loss_list = []

    exp_setting = '-'.join('{}:{}'.format(k, v) for k, v in vars(args).items()
                           if k in args_filter)
    Log = Logger(fn='./logs/{}-{}-{:.0f}.log'.format(model_name, args.dataset,
                                                     start_time))
    Log.log(exp_setting)
    record_manager = RunRecordManager(args.db)
    record_manager.start_run(model_name, start_time, args)

    item_cates = torch.from_numpy(np.array(item_cates)).to(device)  #[all]

    for epoch in range(args.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))

        L = nn.CrossEntropyLoss(
            reduce='none')  # reduce=('none' if args.weight_loss else 'mean')
        _loss = 0

        for i, (bgs, label, next) in enumerate(iters['train']):

            model.train()
            outputs, embeddings, session_length = model.forward(
                bgs.to(device), next.to(device))
            # print(outputs)
            # break
            item_catess = item_cates.view(1, -1).expand_as(outputs)
            mask = torch.where(item_catess == next.to(device),
                               torch.ones_like(item_catess),
                               torch.zeros_like(item_catess))  # [bs,all]
            mask = torch.cat([mask[:, 1:], mask[:, 0].view(-1, 1)], dim=1)
            outputs = outputs * mask  # [bs,all]

            label = label.to(device)
            model.zero_grad()
            y = (label - 1).squeeze()
            # cosine_loss = L_cos(h_all, c_all, target=y)
            loss = L(outputs, y) #- 0.1 * cosine_loss
            # loss_corr=model.corr_loss(embeddings,session_length)
            # # print(loss); print(loss_corr)
            # loss+=loss_corr*args.beta
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()

            if total_batch % STEP_SIZE == 0:
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6},  Time: {2} {3}'
                _loss = np.mean(loss_list)
                Log.log(
                    msg.format(total_batch, _loss, time_dif, '*'))
                loss_list = []
            total_batch += 1

        print('performance on test set....')
        scheduler.step()
        infos = "\n"
        metrics = {}
        for key in iters:
            if key == 'test':
                # acc=0;continue
                acc, info, m = evaluate_topk(args, model, iters[key],
                                             item_cates, device, 20, key)
                metrics[key] = m
                infos += info
            elif key != 'train':
                acc_l, info_l, m_l = evaluate_topk(args,
                                                   model,
                                                   iters[key],
                                                   item_cates,
                                                   device,
                                                   20,
                                                   key,
                                                   observe=False)
                metrics[key] = m_l
                infos += info_l
            infos += "\n"

        msg = f'epoch[{epoch + 1}] :{infos}'
        for test_set_name, m in metrics.items():
            for top_k, v in m.items():
                record_manager.update_best(model_name, start_time,
                                            epoch + 1, test_set_name, top_k,
                                            v['acc'], v['mrr'], v['ndcg'],
                                            _loss)
        if acc > best_acc:
            best_acc = acc
            Log.log(msg, red=True)
            last_improve = 0
            if args.save_flag:
                torch.save(model.state_dict(),
                           './ckpt/{}_epoch{}.ckpt'.format(exp_setting, epoch))

        else:
            Log.log(msg, red=False)
            last_improve += 1
            if last_improve >= args.patience:
                Log.log('Early stop: No more improvement')
                break
        
        
        # try to release gpu memory hold by validation/test set
        # torch.cuda.empty_cache()


def evaluate_topk(args,
                  model,
                  data_iter,
                  item_cates,
                  device,
                  anchor=20,
                  des='',
                  observe=False):
    model.eval()

    res = {'5': [], '10': [], '20': [], '50': []}
    ret_metrics = {}
    labels = []
    acc_anchor = 0
    with torch.no_grad():
        with tqdm(total=(data_iter.__len__()), desc='Predicting',
                  leave=False) as p:
            for i, (bgs, label, next) in (enumerate(data_iter)):
                # print(datas)
                outputs, _, _ = model.forward(bgs.to(device), next.to(device))

                item_catess = item_cates.view(1, -1).expand_as(outputs)
                mask = torch.where(item_catess == next.to(device),
                                   torch.ones_like(item_catess),
                                   torch.zeros_like(item_catess))  # [bs,all]
                mask = torch.cat([mask[:, 1:], mask[:, 0].view(-1, 1)], dim=1)
                outputs = outputs * mask  # [bs,all]

                for k in res:
                    res[k].append(outputs.topk(int(k))[1].cpu())
                labels.append(label)

                p.update(1)
    labels = np.concatenate(labels)  # .flatten()
    labels = labels - 1

    if observe:
        graphs = dgl.unbatch(bgs)
        length = min(20, len(graphs))
        for i in range(length):
            print(graphs[i].nodes['i'].data['id'])
        print(label[0:length])
        sm = outputs.topk(int(20))[1].cpu()[0:length].numpy() + 1
        for i in range(length):
            print(sm[i].tolist())

    print(des)
    msg = des + '\n'
    for k in res:
        acc, mrr, ndcg = metrics(res[k], labels)
        print("Top{} : acc {} , mrr {}, ndcg {}".format(k, acc, mrr, ndcg))
        msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(
            k, acc * 100, mrr * 100, ndcg * 100)
        if int(k) == anchor:
            acc_anchor = acc
        ret_metrics[k] = {'acc': acc, 'mrr': mrr, 'ndcg': ndcg}

    return acc_anchor, msg, ret_metrics


path_check(['./logs', './ckpt'])

argparser = argparse.ArgumentParser('CDSBR')
argparser.add_argument('--model_name', default='IAGNN', type=str, help='model name')
argparser.add_argument('--seed', default=422, type=int, help='random seed')
argparser.add_argument('--emb_size',
                       default=128,
                       type=int,
                       help='embedding size')
argparser.add_argument('--gpu', default=0, type=int, help='gpu id')
# data related setting
argparser.add_argument('--max_length',
                       default=10,
                       type=int,
                       help='max session length')
argparser.add_argument('--dataset',
                       default='jdata_cd',
                       help='dataset=[yc_BT_16|jdata_cd]')
# train related setting
argparser.add_argument('--batch', default=512, type=int, help='batch size')
argparser.add_argument('--epochs', default=10, type=int, help='total epochs')
argparser.add_argument('--patience',
                       default=3,
                       type=int,
                       help='early stopping patience')
argparser.add_argument('--lr', default=0.003, type=float, help='learning rate')
argparser.add_argument('--lr_step', default=3, type=int, help='lr decay step')
argparser.add_argument('--lr_gama',
                       default=0.1,
                       type=float,
                       help='lr decay gama')
argparser.add_argument('--save_flag',
                       default=False,
                       type=bool,
                       help='save checkpoint or not')
argparser.add_argument('--debug',
                       default=False,
                       type=bool,
                       help='cpu mode for debug')
# dropout related setting
argparser.add_argument('--fdrop', default=0.2, type=float, help='feature drop')
argparser.add_argument('--adrop',
                       default=0.0,
                       type=float,
                       help='attention drop')
# model ralated setting
argparser.add_argument('--GL', default=3, type=int, help='gnn layers')
argparser.add_argument('--vinitial', default='id', help='id/mean/max/sum/gru')
argparser.add_argument('--graph_feature_select',
                       default='gated',
                       help='last/gated/mean')
argparser.add_argument('--pooling',
                       default='cnext',
                       help='ilast/imean/cmean/cnext/input')
argparser.add_argument('--cluster_type',
                       default='mean',
                       help='mean/max/last/mean+')
argparser.add_argument('--predictor',
                       default='cosine',
                       help='cosine/bicosine/bilinear/matmul')
argparser.add_argument('--add_loss',
                       default=False,
                       type=bool,
                       help='add corr losss or not')
argparser.add_argument('--beta',
                       default=10.0,
                       type=float,
                       help='corr loss weight')
argparser.add_argument('--tao',
                       default=1.0,
                       type=float,
                       help='weight for softmax')  #需要调参
# model comments
argparser.add_argument('--comment',
                       default='None',
                       type=str,
                       help='other introduction')
argparser.add_argument('--statistics',
                        action='store_true',
                       help='show data statistics')

# record result
argparser.add_argument('--db',
                        default='sqlite',
                        type=str,
                        choices=['sqlite', 'mysql'],
                        help='record the result to sqlite or mysql database.')
args = argparser.parse_args()
print(args)
args_filter = [
    'dataset', 'GL', 'predictor', 'add_loss', 'beta', 'tao'
    'batch', 'lr', 'lr_step', 'emb_size', 'fdrop', 'adrop', 'max_length',
    'comment'
]  # recording hyper-parameters

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available()
                      and args.debug == False and args.gpu >= 0 else 'cpu')

if args.dataset.startswith('yc_BT'):
    data = yoochoose
elif args.dataset.startswith('jd'):
    data = jdata
elif args.dataset.startswith('digi'):
    data = yoochoose

path = '../dataset/'
# modes=["train" ,"test" ,"test_buy" ]
all_data, max_vid, item_cates = data.load_cd_data(
    path + args.dataset, type='aug', test_length=True,
    highfreq_only=True)  # type='aug','common'

if args.statistics:
    data_statistics(all_data)

print(max_vid)
data_describe(dataset=args.dataset, datas=all_data)
set_seed(args.seed)

collate_fn = gnn_collate_fn(seq_to_SSL_graph)

all_data, num_class = datasets(all_data, data.TBVSessionDataset,
                               args.max_length, max_vid)
iters = dataloaders(datas=all_data, batch=args.batch, collate=collate_fn)

model = IAGNN(num_class,
              args.emb_size,
              num_layers=args.GL,
              device=device,
              batch_norm=True,
              add_loss=args.add_loss,
              feat_drop=args.fdrop,
              attention_drop=args.adrop,
              tao=args.tao,
              vinitial_type=args.vinitial,
              graph_feature_select=args.graph_feature_select,
              pooling_type=args.pooling,
              predictor_type=args.predictor).to(device)

optimizer = torch.optim.AdamW(fix_weight_decay(model),
                              lr=args.lr,
                              weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=args.lr_step,
                                            gamma=args.lr_gama)

train(args, model, optimizer, scheduler, device, iters, args_filter,
      item_cates)
