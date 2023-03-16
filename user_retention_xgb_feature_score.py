
#常用工具库
import re
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
import logging.handlers

#算法辅助 & 数据
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from utils_helper import *

#算法（单一学习器）
import xgboost as xgb


#融合模型
from user_retention_xgb import xgb_fit as xgb_fit_feature

"""Train the xgboost model."""

LOG_FILE = 'log/xgb_train.log'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Config(object):
    def __init__(self):
        self.params = {'learning_rate': 0.5,
                       'eval_metric': 'auc',
                       'n_estimators': 5000,
                       'max_depth': 2,
                       'min_child_weight': 5,
                       'gamma': 0,
                       'subsample': 0.8,
                       'colsample_bytree': 0.6,
                       'eta': 0.05,  # 同 learning rate, Shrinkage（缩减），每次迭代完后叶子节点乘以这系数，削弱每棵树的权重
                       'silent': 1,
                    #    'alpha': 1,  # L1  regularization
                       'objective': 'binary:logistic',
                    #    'nthread': 8,
                       'scale_pos_weight': 1}
        self.max_round = 500
        self.cv_folds = 5
        self.early_stop_round = 100
        self.seed = 3
        self.save_model_path = 'model/xgb_user_retention.dat'


def run_cv_fearture(config, X_train, y_train):
    # train model
    tic = time.time()

    xgb_model, best_auc, best_round, cv_result = xgb_fit_feature(config, X_train, y_train)
    print('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
    logger.info(result_message)
    print(result_message)

    # feature analyze
    feature_score_path = 'features/xgb_feature_score.csv'
    feature_analyze(xgb_model, csv_path=feature_score_path, to_print=True, to_plot=True)


if __name__ == "__main__":

    label_file = "data/0403_churn_2.csv"
    df = get_input_df(label_file)
    # split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(['user_id', 'label'], axis=1)
    y_train = train_df['label']
    test_user_id = test_df[['user_id']]
    X_test = test_df.drop(['user_id', 'label'], axis=1)
    y_test = test_df['label']

    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    config = Config()
    run_cv_fearture(config, X_train, y_train)

