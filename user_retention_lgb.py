# -*- coding:utf-8 -*- 

from utils_helper import *

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import time
import logging.handlers

"""Train the lightGBM model."""

LOG_FILE = 'log/lgb_train.log'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Config(object):
    def __init__(self):
        self.params = {
            'objective': 'binary',
            'metric': {'auc'},
            'learning_rate': 0.8,
            'num_leaves': 2,  # 叶子设置为 50 线下过拟合严重
            'min_sum_hessian_in_leaf': 0.1,
            'feature_fraction': 0.3,  # 相当于 colsample_bytree
            'bagging_fraction': 0.5,  # 相当于 subsample
            'lambda_l1': 0,
            'lambda_l2': 5,
            "verbose": -1,  # Set log level to Warning
            'num_thread': 6,  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
            'lambda_l2': 1,
        }
        self.max_round = 500
        self.cv_folds = 5
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = 'model/lgb.txt'


def lgb_fit(config, X_train, y_train):
    """模型（交叉验证）训练，并返回最优迭代次数和最优的结果。
    Args:
        config: xgb 模型参数 {params, max_round, cv_folds, early_stop_round, seed, save_model_path}
        X_train: array like, shape = n_sample * n_feature
        y_train:  shape = n_sample * 1

    Returns:
        best_model: 训练好的最优模型
        best_auc: float, 在测试集上面的 AUC 值。
        best_round: int, 最优迭代次数。
    """
    if type(X_train) == np.ndarray:
        X_train = pd.DataFrame(X_train, columns=features)
        # print(X_train.head(2))
        # apply the column data types to the DataFrame
        X_train = X_train.astype(dtype)
    if type(y_train) == np.ndarray:
        y_train = pd.DataFrame({'label': y_train})
        y_train['label'] = y_train['label'].astype('int8')
    params = config.params
    max_round = config.max_round
    cv_folds = config.cv_folds
    early_stop_round = config.early_stop_round
    seed = config.seed
    # seed = np.random.randint(0, 10000)
    save_model_path = config.save_model_path
    if cv_folds is not None:
        dtrain = lgb.Dataset(X_train, label=y_train)
        cv_result = lgb.cv(params, dtrain, max_round, nfold=cv_folds, seed=seed, verbose_eval=True,
                           metrics='auc', early_stopping_rounds=early_stop_round, show_stdv=False)
        # 最优模型，最优迭代次数
        best_round = len(cv_result['auc-mean'])
        best_auc = cv_result['auc-mean'][-1]  # 最好的 auc 值
        best_model = lgb.train(params, dtrain, best_round)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        watchlist = [dtrain, dvalid]
        best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None
    if save_model_path:
        best_model.save_model(save_model_path)
    return best_model, best_auc, best_round, cv_result


def lgb_predict(model, X_test, user_id=None, save_result_path=None):
    if type(X_test) == np.ndarray:
        X_test = pd.DataFrame(X_test, columns=features)
        # apply the column data types to the DataFrame
        X_test = X_test.astype(dtype)
    y_pred_prob = model.predict(X_test)
    if save_result_path:
        df_result = user_id
        df_result['churn'] = y_pred_prob
        df_result.to_csv(save_result_path, index=False)
        print('Save the result to {}'.format(save_result_path))
    return y_pred_prob


def run_cv(config, X_train, X_test, y_train, user_id=None):
    config = Config()
    # train model
    tic = time.time()

    lgb_model, best_auc, best_round, cv_result = lgb_fit(config, X_train, y_train)
    print('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
    logger.info(result_message)
    print(result_message)
    # predict
    # lgb_model = lgb.Booster(model_file=config.save_model_path)
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_auc)
    result = lgb_predict(lgb_model, X_test, user_id, result_path)

    return result


if __name__ == '__main__':
    # get feature
    label_file = "data/0403_churn_2.csv"
    df = get_input_df(label_file)
    # split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(['user_id','label'], axis=1)
    y_train = train_df['label']
    test_user_id = test_df[['user_id']]
    X_test = test_df.drop(['user_id', 'label'], axis=1)
    y_test = test_df['label']

    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    config = Config()
    test_pred = run_cv(config, X_train, X_test, y_train, test_user_id)
    message = 'Test ROC AUC:', roc_auc_score(y_test, test_pred)
    logger.info(message)
    print(message)

