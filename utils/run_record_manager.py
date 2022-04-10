# coding=utf-8
# Main functions:
# 实现指标查找、指标排序、附加文件打开

from ast import arg
from typing import Dict
import json
import os
import sqlite3
import pymysql
from pymysql.cursors import DictCursor

RUN_ARGS = [
    'seed',
    'emb_size',
    'gpu',
    'max_length',
    'dataset',
    'batch',
    'epochs',
    'patience',
    'lr',
    'lr_step',
    'lr_gama',
    'save_flag',
    'debug',
    'fdrop',
    'adrop',
    'GL',
    'vinitial',
    'graph_feature_select',
    'pooling',
    'cluster_type',
    'predictor',
    'add_loss',
    'beta',
    'tao',
    'comment'
]

DB_CREATION_SCRIPT = '''
CREATE TABLE IF NOT EXISTS run (
	MODEL TEXT(255),
	TIMESTAMP REAL,
	seed INTEGER,
	emb_size INTEGER,
	gpu INTEGER,
	max_length INTEGER,
	dataset TEXT,
	batch INTEGER,
	epochs INTEGER,
	patience INTEGER,
	lr REAL,
	lr_step INTEGER,
	lr_gama REAL,
	save_flag INTEGER,
	debug INTEGER,
	fdrop REAL,
	adrop REAL,
	GL INTEGER,
	vinitial TEXT,
	graph_feature_select TEXT,
	pooling TEXT,
	cluster_type TEXT,
	predictor TEXT,
	add_loss INTEGER,
	beta REAL,
    tao REAL,
	comment TEXT,
	CONSTRAINT run_PK PRIMARY KEY (MODEL, TIMESTAMP)
);

CREATE TABLE IF NOT EXISTS record (
	MODEL TEXT(255),
	TIMESTAMP REAL,
	EPOCH INTEGER,
	TEST_SET TEXT(255),
	TOP_K INTEGER,
	acc REAL,
	mrr REAL,
	ndcg REAL,
    train_loss REAL,
    is_deleted INTEGER,
	CONSTRAINT record_PK PRIMARY KEY (MODEL, TIMESTAMP, EPOCH, TEST_SET, TOP_K)
);
'''

class IDBProvider:
    def init_db(self):
        raise NotImplementedError()
    
    def start_run(self,
                  model_name: str,
                  timestamp: int,
                  args: dict,
                  ):
        raise NotImplementedError()
    
    def update_best(self,
                model_name: str,
                timestamp: int,
                epoch: int,
                test_set: str,
                top_k: int,
                acc: float,
                mrr: float,
                ndcg: float,
                train_loss: float
                ):
        raise NotImplementedError()

class SQLiteProvider(IDBProvider):
    conn: sqlite3.Connection = sqlite3.connect('../runs.db', timeout=30000)

    def __init__(self) -> None:
        # create tables if not exists
        self.init_db()
        pass

    def init_db(self):
        with self.conn:
            cur: sqlite3.Cursor = self.conn.executescript(DB_CREATION_SCRIPT)

    def start_run(self,
                  model_name: str,
                  timestamp: int,
                  args: dict,
                  ):
        with self.conn:
            cur: sqlite3.Cursor = self.conn.execute(
                '''
REPLACE INTO run(
    MODEL,
	TIMESTAMP,
	{}
) VALUES ({})
'''.format(','.join(RUN_ARGS), ','.join(['?'] * 27)),
                (model_name, timestamp) +
                tuple([vars(args).get(a) for a in RUN_ARGS])
            )

    def update_best(self,
                    model_name: str,
                    timestamp: int,
                    epoch: int,
                    test_set: str,
                    top_k: int,
                    acc: float,
                    mrr: float,
                    ndcg: float,
                    train_loss: float
                    ):
        with self.conn:
            cur: sqlite3.Cursor = self.conn.execute(
'''
REPLACE INTO record(MODEL, TIMESTAMP, EPOCH, TEST_SET, TOP_K, acc, mrr, ndcg, train_loss)
VALUES({})
'''.format(','.join(['?'] * 9)), 
                (model_name, timestamp, epoch, test_set, top_k, acc, mrr, ndcg, train_loss)
            )

class MySqlProvider(IDBProvider):
    
    conn: pymysql.Connection

    def __init__(self) -> None:
        super().__init__()
        with open('../mysql.conf', encoding='utf8') as f:
            mysql_conf: Dict = json.load(f)
            self.conn = pymysql.connect(host=mysql_conf['host'],
                             user=mysql_conf['user'],
                             password=mysql_conf['password'],
                             database=mysql_conf['database'],
                             charset=mysql_conf['charset'],
                             port=mysql_conf['port'],
                             cursorclass=pymysql.cursors.DictCursor)
            self.conn.autocommit(True)



    def init_db(self):
        raise NotImplementedError()
    
    def start_run(self,
                  model_name: str,
                  timestamp: int,
                  args: dict,
                  ):
        self.conn.ping()
        with self.conn.cursor() as cur:
            cur: DictCursor = cur
            cur.execute(
                '''
REPLACE INTO run(
    MODEL,
	TIMESTAMP,
	{}
) VALUES ({})
'''.format(','.join(RUN_ARGS), ','.join(['%s'] * 27)),
                (model_name, timestamp) +
                tuple([vars(args).get(a) for a in RUN_ARGS])
            )
    
    def update_best(self,
                model_name: str,
                timestamp: int,
                epoch: int,
                test_set: str,
                top_k: int,
                acc: float,
                mrr: float,
                ndcg: float,
                train_loss: float
                ):
        self.conn.ping()
        with self.conn.cursor() as cur:
            cur: DictCursor = cur
            cur.execute(
'''
REPLACE INTO record(MODEL, TIMESTAMP, EPOCH, TEST_SET, TOP_K, acc, mrr, ndcg, train_loss)
VALUES({})
'''.format(','.join(['%s'] * 9)), 
                (model_name, timestamp, epoch, test_set, top_k, acc, mrr, ndcg, float(train_loss))
            )
class RunRecordManager(IDBProvider):
    _provider: IDBProvider

    def __init__(self, provider: str = 'sqlite') -> None:
        super().__init__()
        provider = provider.lower()
        if provider == 'sqlite':
            self._provider = SQLiteProvider()
        elif provider == 'mysql':
            self._provider = MySqlProvider()
    
    
    
    def start_run(self,
                  model_name: str,
                  timestamp: int,
                  args: dict,
                  ):
        self._provider.start_run(model_name, timestamp, args)
    
    def update_best(self,
                model_name: str,
                timestamp: int,
                epoch: int,
                test_set: str,
                top_k: int,
                acc: float,
                mrr: float,
                ndcg: float,
                train_loss: float
                ):
        self._provider.update_best(
                model_name,
                timestamp,
                epoch,
                test_set,
                top_k,
                acc,
                mrr,
                ndcg,
                train_loss
        )

# leaderboard?
# select * from record left join run on record.MODEL = run.MODEL and record."TIMESTAMP" = run."TIMESTAMP" where TEST_SET = 'test' and TOP_K = 20 order by mrr;
