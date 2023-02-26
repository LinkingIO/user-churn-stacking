
#常用工具库
import re
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
import logging.handlers

#算法辅助 & 数据
import sklearn
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error
import joblib
from utils_helper import *


#算法（单一学习器）
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LogiR
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

#融合模型
from sklearn.ensemble import StackingClassifier

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
        self.params = {'learning_rate': 0.05,
                       'eval_metric': 'auc',
                       'n_estimators': 5000,
                       'max_depth': 6,
                       'min_child_weight': 7,
                       'gamma': 0,
                       'subsample': 0.8,
                       'colsample_bytree': 0.6,
                       'eta': 0.05,  # 同 learning rate, Shrinkage（缩减），每次迭代完后叶子节点乘以这系数，削弱每棵树的权重
                       'silent': 1,
                       'objective': 'binary:logistic',
                    #    'nthread': 8,
                       'scale_pos_weight': 1}
        self.max_round = 1000
        self.cv_folds = 5
        self.early_stop_round = 200
        self.seed = 3
        self.save_model_path = 'model/xgb_user_retention.dat'


def xgb_fit(config, X_train, y_train):
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
    save_model_path = config.save_model_path

    if cv_folds is not None:
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        cv_result = xgb.cv(params, dtrain, max_round, nfold=cv_folds, seed=seed, verbose_eval=True,
                           metrics='auc', early_stopping_rounds=early_stop_round, show_stdv=False)
        # 最优模型，最优迭代次数
        best_round = cv_result.shape[0]
        best_auc = cv_result['test-auc-mean'].values[-1]  # 最好的 auc 值
        best_model = xgb.train(params, dtrain, best_round)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        best_model = xgb.train(params, dtrain, max_round, evals=watchlist, early_stopping_rounds=early_stop_round, verbosity=0)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None
    if save_model_path:
        joblib.dump(best_model, save_model_path)
    return best_model, best_auc, best_round, cv_result


def xgb_predict(model, X_test, save_result_path=None):
    if type(X_test) == np.ndarray:
        X_test = pd.DataFrame(X_test, columns=features)
        # apply the column data types to the DataFrame
        X_test = X_test.astype(dtype)
    user_id = X_test[['user_id']]
    X_test = X_test.drop(['user_id'], axis=1)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    y_pred_prob = model.predict(dtest)
    if save_result_path:
        df_result = user_id
        df_result['churn'] = y_pred_prob
        df_result.to_csv(save_result_path, index=False)
        print('Save the result to {}'.format(save_result_path))
    return y_pred_prob


def run_cv(config, X_train, y_train, X_test):
    # train model
    tic = time.time()

    xgb_model, best_auc, best_round, cv_result = xgb_fit(config, X_train, y_train)
    print('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
    logger.info(result_message)
    print(result_message)

    # predict
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_xgb_{}-{:.4f}.csv'.format(now, best_auc)
    result = xgb_predict(xgb_model, X_test, result_path)

    # feature analyze
    feature_score_path = 'features/xgb_feature_score.csv'
    feature_analyze(xgb_model, csv_path=feature_score_path)

    return result


if __name__ == "__main__":

    label_file = "data/0403_churn_2.csv"
    df = get_input_df(label_file)
    # split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(['user_id', 'label'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label'], axis=1)
    y_test = test_df['label']

    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    config = Config()

    run_cv(config, X_train, y_train, X_test)

