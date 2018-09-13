# -*- coding=utf-8 -*-
import gc
import time
import pandas as pd
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBClassifier
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

time_begin = time.time()

import getopt, sys

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:")
except getopt.GetoptError:
    print('parameters: xxxxx.py -t <target>')
    sys.exit(2)

target = ''
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for (i, j) in opts:
    if i == '-t' and j in class_names:
        target = j
    else:
        print("Parameters error!")
        sys.exit(2)

# ---------------读取数据---------------------

train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')
test_feats = pd.read_csv('../../output/v1/test_feats.csv').fillna(' ')

submission = pd.DataFrame.from_dict({'id': test_feats['id']})
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def modelfit(model, X_train, y_train, useTrainCV=True, cv_fold=3, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        print('交叉验证自动设定迭代次数...')
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=xgb_param['n_estimators'],
                          nfold=cv_fold,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          seed=36,
                          callbacks=[xgb.callback.print_evaluation(period=10, show_stdv=True)])
        print('cv完成 n_estimators：', cvresult.shape[0])
        model.set_params(n_estimators=cvresult.shape[0])
    # fit
    print('模型拟合中...')
    model.fit(X_train,
              y_train,
              eval_metric='auc')


nthread = 3


def get_tuned_xgb(target):
    if target == 'toxic':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=500,
            scale_pos_weight=0.5,
            max_depth=5,
            min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            objective='binary:logistic',
            nthread=nthread,
            seed=36)
    elif target == 'severe_toxic':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=145,
            scale_pos_weight=0.7,
            max_depth=3,
            min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            objective='binary:logistic',
            nthread=nthread,
            seed=36)
    elif target == 'obscene':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=279,
            scale_pos_weight=0.7,
            max_depth=10,
            min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=nthread,
            seed=36)
    elif target == 'threat':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=100,
            scale_pos_weight=0.7,
            max_depth=10,
            min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=nthread,
            seed=36)
    elif target == 'insult':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=500,
            scale_pos_weight=0.5,
            max_depth=10,
            min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=nthread,
            seed=36)
    elif target == 'identity_hate':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=127,
            scale_pos_weight=0.5,
            max_depth=10,
            min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=nthread,
            seed=36)


feat_importaces = pd.DataFrame()

# =============== single model ===============
# for target in class_names:
#     xgb_model = get_tuned_xgb(target)
#     train_target = train_data[target]
#
#     print('Loading data')
#     train_sparse_matrix = load_npz("../../output/v2/xgb_train_less_"+target+".npz")
#     test_sparse_matrix = load_npz("../../output/v2/xgb_test_less_"+target+".npz")
#     print("Start training: ", target)
#     modelfit(xgb_model,
#              X_train=train_sparse_matrix,
#              y_train=train_data[target],
#              useTrainCV=True)
#     submission[target] = xgb_model.predict_proba(test_sparse_matrix)[:, 1]
#     feat_importaces[target] = pd.Series(xgb_model.feature_importances_)
# submission.to_csv('../../output/v1/xgb_submission.csv', index=False)
# feat_importaces.to_csv('../../output/v1/xgb_feat_imp.csv', index=False)


# =================格点搜索================

xgb_model = get_tuned_xgb(target)
train_target = train_data[target]

print('Loading data')
train_sparse_matrix = load_npz("../../output/v2/xgb_train_less_" + target + ".npz")
test_sparse_matrix = load_npz("../../output/v2/xgb_test_less_" + target + ".npz")

print("tuning: ", target)
# 格点搜索参数
param_test = {}
# if target == 'toxic':
#     param_test = {
#         'min_child_weight': [3, 5, 7]  # 1
#     }
# elif target == 'severe_toxic':
#     param_test = {
#         'min_child_weight': [3, 5, 7]  # 1
#     }
# else:
#     param_test = {
#         'max_depth': [15, 20]  # 10
#     }

gsearch = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_test,
    scoring='roc_auc',
    iid=False,
    cv=3,
    n_jobs=3,
)

# print('正在搜索...')
gsearch.fit(train_sparse_matrix, train_data[target])

# 输出搜索结果
print('\n当前格点搜索的参数：\n', param_test)
print('\ngsearch.best_params_:', gsearch.best_params_, )
print('\ngsearch.best_score_:', gsearch.best_score_, )

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
