# -*- coding=utf-8 -*-

# 用调参后的模型生成用于第二层的stacking特征

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import getpass
from sklearn.metrics import roc_auc_score, mean_absolute_error, auc
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import time
from scipy.sparse import load_npz, save_npz

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
print('Start: ', target)

train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')
test_feats = pd.read_csv('../../output/v1/test_feats.csv').fillna(' ')

# train_sparse_matrix = load_npz("../../output/v1_train/lr_train_feats.npz").tocsr()
# test_sparse_matrix = load_npz("../../output/v1_train/lr_test_feats.npz").tocsr()

train_sparse_matrix = load_npz("../../output/v3/xgb_exchanged_train_less_" + target + ".npz")
test_sparse_matrix = load_npz("../../output/v3/xgb_exchanged_test_less_" + target + ".npz")

print('read file finished')

ntrain = train_data.shape[0]
ntest = test_feats.shape[0]

x_train = train_sparse_matrix
y_train = train_data[target]
x_test = test_sparse_matrix

x_train = train_sparse_matrix[:100]
y_train = train_data[target][:100]
x_test = test_sparse_matrix[:100]
ntrain = 100
ntest = 100

kf = KFold(n_splits=5, shuffle=True, random_state=36)


class Best_models:
    def get_tuned_xgb(self, target):
        if target == 'toxic':
            return XGBClassifier(
                learning_rate=0.1,  # eta
                n_estimators=1000,
                scale_pos_weight=0.5,
                max_depth=5,
                min_child_weight=1,
                objective='binary:logistic',
                nthread=-1,
                seed=36)
        elif target == 'severe_toxic':
            return XGBClassifier(
                learning_rate=0.1,  # eta
                n_estimators=290,
                scale_pos_weight=0.7,
                max_depth=3,
                min_child_weight=1,
                objective='binary:logistic',
                nthread=-1,
                seed=36)
        elif target == 'obscene':
            return XGBClassifier(
                learning_rate=0.1,  # eta
                n_estimators=279 * 2,
                scale_pos_weight=0.7,
                max_depth=10,
                min_child_weight=1,
                objective='binary:logistic',
                nthread=-1,
                seed=36)
        elif target == 'threat':
            return XGBClassifier(
                learning_rate=0.1,  # eta
                n_estimators=200,
                scale_pos_weight=0.7,
                max_depth=10,
                min_child_weight=1,
                objective='binary:logistic',
                nthread=-1,
                seed=36)
        elif target == 'insult':
            return XGBClassifier(
                learning_rate=0.1,  # eta
                n_estimators=1000,
                scale_pos_weight=0.5,
                max_depth=10,
                min_child_weight=1,
                objective='binary:logistic',
                nthread=-1,
                seed=36)
        elif target == 'identity_hate':
            return XGBClassifier(
                learning_rate=0.1,  # eta
                n_estimators=127 * 2,
                scale_pos_weight=0.5,
                max_depth=10,
                min_child_weight=1,
                objective='binary:logistic',
                nthread=-1,
                seed=36)

    def get_tuned_lgb(self, target):
        if target == 'toxic':
            return LGBMClassifier(
                learning_rate=0.1,
                n_estimators=442 * 2,
                num_leaves=100,
                max_depth=-1,
                min_split_gain=0.,
                min_child_weight=1e-3,
                min_child_samples=20,
                subsample=1.,
                subsample_freq=1,
                colsample_bytree=1.,
                reg_alpha=0.,
                reg_lambda=0.,
                random_state=36,
                n_jobs=-1,
                silent=False, )
        elif target == 'severe_toxic':
            return LGBMClassifier(
                learning_rate=0.1,
                n_estimators=83 * 2,
                num_leaves=100,
                max_depth=-1,
                min_split_gain=0.,
                min_child_weight=1e-3,
                min_child_samples=20,
                subsample=1.,
                subsample_freq=1,
                colsample_bytree=1.,
                reg_alpha=0.,
                reg_lambda=0.,
                random_state=36,
                n_jobs=-1,
                silent=False, )
        elif target == 'obscene':
            return LGBMClassifier(
                learning_rate=0.1,
                n_estimators=135 * 2,
                num_leaves=63,
                max_depth=-1,
                min_split_gain=0.,
                min_child_weight=1e-3,
                min_child_samples=20,
                subsample=1.,
                subsample_freq=1,
                colsample_bytree=1.,
                reg_alpha=0.,
                reg_lambda=0.,
                random_state=36,
                n_jobs=-1,
                silent=False, )
        elif target == 'threat':
            return LGBMClassifier(
                learning_rate=0.1,
                n_estimators=126 * 2,
                num_leaves=31,
                max_depth=-1,
                min_split_gain=0.,
                min_child_weight=1e-3,
                min_child_samples=20,
                subsample=1.,
                subsample_freq=1,
                colsample_bytree=1.,
                reg_alpha=0.,
                reg_lambda=0.,
                random_state=36,
                n_jobs=-1,
                silent=False, )
        elif target == 'insult':
            return LGBMClassifier(
                learning_rate=0.1,
                n_estimators=367 * 2,
                num_leaves=100,
                max_depth=-1,
                min_split_gain=0.,
                min_child_weight=1e-3,
                min_child_samples=20,
                subsample=1.,
                subsample_freq=1,
                colsample_bytree=1.,
                reg_alpha=0.,
                reg_lambda=0.,
                random_state=36,
                n_jobs=-1,
                silent=False, )
        elif target == 'identity_hate':
            return LGBMClassifier(
                learning_rate=0.1,
                n_estimators=119 * 2,
                num_leaves=100,
                max_depth=-1,
                min_split_gain=0.,
                min_child_weight=1e-3,
                min_child_samples=20,
                subsample=1.,
                subsample_freq=1,
                colsample_bytree=1.,
                reg_alpha=0.,
                reg_lambda=0.,
                random_state=36,
                n_jobs=-1,
                silent=False, )

    def get_tuned_lr(self):
        return LogisticRegression(max_iter=300,
                                  penalty='l2',
                                  C=1,
                                  class_weight='balanced',
                                  solver='sag', )


bm = Best_models()


def get_oof(model, model_name):
    print("get_oof")
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        print("enumerate ", i)
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        kf_x_test = x_train[test_index]

        print(model_name, 'trainning...　数据量:{},{}'.format(kf_x_train.shape, kf_y_train.shape))
        model.fit(kf_x_train, kf_y_train)

        oof_train[test_index] = model.predict_proba(kf_x_test)[:, 1]
        oof_test_skf[i, :] = model.predict_proba(x_test)[:, 1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# 初始化各个调过参数的模型
xgb_model = bm.get_tuned_xgb(target)
# lgb_model = bm.get_tuned_lgb(target)
# lr_model = bm.get_tuned_lr()

# 产生在训练集上交叉预测的列，以及在测试集上预测的平均值
xgb_oof_train, xgb_oof_test = get_oof(xgb_model, 'XGB')
# lgb_oof_train, lgb_oof_test = get_oof(lgb_model, 'LGB')
# lr_oof_train, lr_oof_test = get_oof(lr_model, 'LR')

# 输出训练集上交叉验证的相关指标
print('\nXGB-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, xgb_oof_train)))
print('XGB-CV roc_auc_score: {}'.format(roc_auc_score(y_train, xgb_oof_train)))

# print('\nLGB-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, lgb_oof_train)))
# print('LGB-CV roc_auc_score: {}'.format(roc_auc_score(y_train, lgb_oof_train)))
#
# print('\nLR-CV mean_absolute_error: {}'.format(mean_absolute_error(y_train, lr_oof_train)))
# print('LR-CV roc_auc_score: {}'.format(roc_auc_score(y_train, lr_oof_train)))

# 产生新的训练集和测试集，即各个算法在训练集上交叉预测的列的并排
z_train = np.concatenate((xgb_oof_train,), axis=1)
z_test = np.concatenate((xgb_oof_test,), axis=1)

print("\nz_train:{}, z_test:{}".format(z_train.shape, z_test.shape))

# 保存新的训练集和测试集
z_train_pd = pd.DataFrame(z_train, columns=['XGB_EX_' + target])
z_test_pd = pd.DataFrame(z_test, columns=['XGB_EX_' + target])
z_train_pd.to_csv('../../output/v3/z_train_xgb_ex_' + target + '.csv', encoding='gbk', index=False)
z_test_pd.to_csv('../../output/v3/z_test_xgb_ex_' + target + '.csv', encoding='gbk', index=False)

# ------------输出运行时间　不需要改---------------
time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
