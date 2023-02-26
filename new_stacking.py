import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

from user_retention_lgb import lgb_fit, lgb_predict
from user_retention_lgb import Config as LGB_Config
from utils_helper import *

import logging.handlers
import time

LOG_FILE = 'log/stacking.log'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('stack')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # Generate  data
    label_file = "data/0403_churn_2.csv"
    df = get_input_df(label_file)
    # split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(['user_id', 'label'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label'], axis=1)
    y_test = test_df['label']
    print("----------------info-----------------------")

    # Create base models
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Define Logistic Regression meta-model
    lr_model = LogisticRegression()

    # Define XGBoost and LightGBM params
    xgb_params = {'learning_rate': 0.05,
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
                    'scale_pos_weight': 1}
    
    lgb_config = LGB_Config()

    # Create arrays to hold predictions from base models
    xgb_train_pred = np.zeros(len(X_train))
    lgb_train_pred = np.zeros(len(X_train))
    xgb_test_pred = np.zeros(len(X_test))
    lgb_test_pred = np.zeros(len(X_test))

    X_test = X_test.drop(['user_id'], axis=1)

    for fold_, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        print(f'Fold {fold_+1}:')
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        lgb_train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        lgb_valid_data = lgb.Dataset(X_val_fold, label=y_val_fold)
        lgb_test_data = lgb.Dataset(X_test, label=y_test)

        # Fit and predict with LightGBM model
        watchlist = [lgb_train_data, lgb_valid_data]
        lgb_model = lgb.train(lgb_config.params, lgb_train_data, lgb_config.max_round, valid_sets=watchlist, early_stopping_rounds=lgb_config.early_stop_round)
        
        result = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)#对验证集得到预测结果
        lgb_test_pred += lgb_model.predict(X_test, ntree_limit=lgb_model.best_iteration) / num_folds

        # xgb train and pred
        xgb_train_data = xgb.DMatrix(X_train_fold, label=y_train_fold, enable_categorical=True)
        xgb_valid_data = xgb.DMatrix(X_val_fold, label=y_val_fold, enable_categorical=True)
        xgb_test_data = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        watchlist = [(xgb_train_data, 'train'), (xgb_valid_data, 'valid_data')]
        xgb_model = xgb.train(dtrain=xgb_train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)#80%用于训练过程

        xgb_train_pred[val_index] = xgb_model.predict(xgb_valid_data, ntree_limit=xgb_model.best_ntree_limit)#预测20%的验证集
        xgb_test_pred += xgb_model.predict(xgb_test_data, ntree_limit=xgb_model.best_ntree_limit) / num_folds

    # Create meta-feature dataframes
    train_meta = pd.DataFrame({'XGB': xgb_train_pred, 'LGB': lgb_train_pred})
    test_meta = pd.DataFrame({'XGB': xgb_test_pred, 'LGB': lgb_test_pred})

    # Fit meta-model on meta-features and make predictions
    lr_model.fit(train_meta, y_train)
    train_pred = lr_model.predict_proba(train_meta)[:,1]
    test_pred = lr_model.predict_proba(test_meta)[:,1]

    # Print ROC AUC scores
    print('Train ROC AUC:', roc_auc_score(y_train, train_pred))
    print('Test ROC AUC:', roc_auc_score(y_test, test_pred))