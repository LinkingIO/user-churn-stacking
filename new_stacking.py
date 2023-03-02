import numpy as np
import pickle
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from user_retention_lgb import lgb_fit, lgb_predict
from user_retention_lgb import Config as LGB_Config
from user_retention_xgb import xgb_fit, xgb_predict
from user_retention_xgb import Config as XGB_Config
from user_retention_cgb import cgb_fit, cgb_predict
from user_retention_cgb import Config as CGB_Config
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
    test_user_id = test_df[['user_id']]
    X_test = test_df.drop(['user_id','label'], axis=1)
    y_test = test_df['label']
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    # Create base models
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Define Logistic Regression meta-model
    param_grids = {
        "C": list(np.linspace(0.0001, 10, 100))
    }
    # lr_model = LogisticRegression(max_iter=300, n_jobs=6)
    grid = GridSearchCV(LogisticRegression(penalty='l2', max_iter=100), param_grid=param_grids, cv=5, scoring="roc_auc")

    # config object
    lgb_config = LGB_Config()
    xgb_config = XGB_Config()
    cgb_config = CGB_Config()

    # Create arrays to hold predictions from base models
    xgb_train_pred = np.zeros(len(X_train))
    lgb_train_pred = np.zeros(len(X_train))
    cgb_train_pred = np.zeros(len(X_train))
    xgb_test_pred = np.zeros(len(X_test))
    lgb_test_pred = np.zeros(len(X_test))
    cgb_test_pred = np.zeros(len(X_test))

    tic = time.time()

    for fold_, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        print(f'Fold {fold_+1}:')
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # #新增随机多样性，相同的算法更换随机数种子
        # clf1 = RFC(n_estimators= 100,max_features="sqrt",max_samples=0.9, random_state=4869,n_jobs=8)
        # estimators = ("RandomForest",clf1)

        # cv = KFold(n_splits=5,shuffle=True,random_state=1412)
        # cv_results = cross_validate(estimator[1],X_train, y_train
        #                      ,cv = cv
        #                      ,scoring = scoring
        #                      ,n_jobs = -1
        #                      ,return_train_score = True
        #                      ,verbose=False)

        # Fit and predict with LightGBM model
        lgb_model, best_auc, best_round, cv_result = lgb_fit(lgb_config, X_train_fold, y_train_fold)
        print('Time cost {}s'.format(time.time() - tic))
        result_message = 'LightGBM fished fold {}/{}, best_round={}, best_auc={}'.format(fold_ + 1, num_folds, best_round, best_auc)
        logger.info(result_message)
        print(result_message)

        xgb_train_pred[val_index] = lgb_predict(lgb_model, X_val_fold)
        now = time.strftime("%m%d-%H%M%S")
        result_path = 'result/result_lgb_test_{}-{:.4f}.csv'.format(now, best_auc)
        lgb_test_pred += lgb_predict(lgb_model, X_test, test_user_id, result_path) / num_folds

        # Fit and predict with xgboost model
        xgb_model, best_auc, best_round, cv_result = xgb_fit(xgb_config, X_train_fold, y_train_fold)
        print('Time cost {}s'.format(time.time() - tic))
        result_message = 'XGBoost fished fold {}/{}, best_round={}, best_auc={}'.format(fold_ + 1, num_folds, best_round, best_auc)
        logger.info(result_message)
        print(result_message)

        # predict
        xgb_train_pred[val_index] = xgb_predict(xgb_model, X_val_fold)
        now = time.strftime("%m%d-%H%M%S")
        result_path = 'result/result_xgb_{}-{:.4f}.csv'.format(now, best_auc)
        xgb_test_pred += xgb_predict(xgb_model, X_test, test_user_id, result_path) / num_folds

        # feature analyze
        feature_score_path = 'features/xgb_feature_score.csv'
        feature_analyze(xgb_model, csv_path=feature_score_path)

        # Fit and predict with CatBoost model
        cgb_model, best_auc, best_round, cv_result = cgb_fit(cgb_config, X_train_fold, y_train_fold)
        print('Time cost {}s'.format(time.time() - tic))
        result_message = 'CatBoost fished fold {}/{}, best_round={}, best_auc={}'.format(fold_ + 1, num_folds, best_round, best_auc)
        logger.info(result_message)
        print(result_message)

        # predict
        cgb_model = pickle.load(open(cgb_config.save_model_path, 'rb'))
        now = time.strftime("%m%d-%H%M%S")
        cgb_train_pred[val_index] = cgb_predict(cgb_model, X_val_fold)
        result_path = 'result/result_cgb_{}.csv'.format(now)
        xgb_test_pred += cgb_predict(cgb_model, X_test, test_user_id,result_path) / num_folds

    # Create meta-feature dataframes
    train_meta = pd.DataFrame({'XGB': xgb_train_pred, 'LGB': lgb_train_pred, 'CGB': cgb_train_pred})
    test_meta = pd.DataFrame({'XGB': xgb_test_pred, 'LGB': lgb_test_pred, 'CGB': cgb_test_pred})
    # train_meta = pd.DataFrame({'XGB': xgb_train_pred, 'CGB': cgb_train_pred})
    # test_meta = pd.DataFrame({'XGB': xgb_test_pred, 'CGB': cgb_test_pred})

    # Fit meta-model on meta-features and make predictions
    grid.fit(train_meta, y_train)
    train_pred = grid.predict_proba(train_meta)[:,1]
    test_pred = grid.predict_proba(test_meta)[:,1]

    # Print ROC AUC scores
    message = 'Train ROC AUC:', roc_auc_score(y_train, train_pred)
    print(message)
    logger.info(message)
    message = 'Test ROC AUC:', roc_auc_score(y_test, test_pred)
    logger.info(message)
    print(message)
