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
from sklearn.metrics import roc_auc_score
time_begin = time.time()

import getopt, sys
try:
    opts, args = getopt.getopt(sys.argv[1:], "t:")
except getopt.GetoptError:
    print('parameters: xgb_gpu.py -t <target>')
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

train_features = load_npz("../../output/v1_train/xgb_train_feats.npz")
test_features = load_npz("../../output/v1_train/xgb_test_feats.npz")

print('Loaded')

submission = pd.DataFrame.from_dict({'id': test_feats['id']})

xgb_param = {"learning_rate": 0.2,
             'max_depth': 3,
             "min_child_weight": 1,
             "gamma": 0,
             'subsample': 1,
             'colsample_bytree': 1,
             'objective': 'binary:logistic',
             "scale_pos_weight": 1,
             'min_child_weight': 1,
             'reg_alpha': 0,
             'reg_lambda': 1,
             'eval_metric': 'auc',
             "tree_method": 'gpu_exact',
             "predictor": 'gpu_predictor',
             "seed": 36}

boost_rounds = {'toxic': 500,
                'severe_toxic': 109,
                'obscene': 248,
                'threat': 146,
                'insult': 433,
                'identity_hate': 172}

feat_importaces = pd.DataFrame()

# =============== single model ===============

train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = \
    train_test_split(train_features, train_data[target], test_size=0.05, random_state=36)

d_train = xgb.DMatrix(train_sparse_matrix, label=y_train)

print("train: ", target)
model = xgb.train(xgb_param,
                  dtrain=d_train,
                  num_boost_round=boost_rounds[target])

test_dmatrix = xgb.DMatrix(test_features)
copy_model = model.copy()
submission[target] = copy_model.predict(test_dmatrix)

valid_dmatrix = xgb.DMatrix(valid_sparse_matrix)
print("Valid Roc:", roc_auc_score(y_valid, copy_model.predict(valid_dmatrix)))


submission.to_csv('../../output/v2/xgb_lrdata_'+target+'__submission.csv', index=False)

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))

