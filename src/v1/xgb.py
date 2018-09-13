# -*- coding=utf-8 -*-
import gc
import time
import pandas as pd
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBClassifier
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV

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

# =================== tfidf ======================

# train_text = train_data['comment_text']
# test_text = test_feats['comment_text']
# all_text = pd.concat([train_text, test_text])
#
# train_clean_text = train_data['clean_text']
# test_clean_text = test_feats['clean_text']
# all_clean_text = pd.concat([train_clean_text, test_clean_text])
#
# word_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='word',
#     token_pattern=r'\w{1,}',
#     ngram_range=(1, 2),
#     max_features=50000)
# word_vectorizer.fit(all_clean_text)
# train_word_features = word_vectorizer.transform(train_clean_text)
# test_word_features = word_vectorizer.transform(test_clean_text)
# print('Word TFIDF finished.')
#
# char_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='char',
#     stop_words='english',
#     ngram_range=(2, 6),
#     max_features=50000)
# char_vectorizer.fit(all_text)
# train_char_features = char_vectorizer.transform(train_text)
# test_char_features = char_vectorizer.transform(test_text)
# print('Char TFIDF finished.')
#
# # ================= add features ==================
#
# NUM_FEATS = ['count_sent', 'count_word', 'count_unique_word',
#              'count_letters', 'count_punctuations', 'count_words_upper',
#              'count_words_title', 'count_stopwords', 'count_!',
#              'count_duplicate', 'count_space', 'max_duplicate_time',
#              'max_upper_len', 'max_word_len', 'percent_max_duplicate',
#              'percent_upper', 'percent_title', 'percent_!',
#              'percent_?', 'percent_number', 'percent_word_unique',
#              'mean_word_len', 'mean_unique_word', 'mean_upper', 'mean_title', ]
#
# train_features = hstack([train_char_features, train_word_features, train_data[NUM_FEATS]])
# test_features = hstack([test_char_features, test_word_features, test_feats[NUM_FEATS]])
# print('HStack finished')
#
# save_npz("../../output/v1_train/xgb_train_feats.npz", train_features)
# save_npz("../../output/v1_train/xgb_test_feats.npz", test_features)
#
# del char_vectorizer
# del word_vectorizer
# del train_char_features
# del train_word_features
# del test_char_features
# del test_word_features
# gc.collect()

train_features = load_npz("../../output/v1_train/xgb_train_feats.npz")
test_features = load_npz("../../output/v1_train/xgb_test_feats.npz")

print('Loaded')

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
                          callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
        print('cv完成 n_estimators：', cvresult.shape[0])
        model.set_params(n_estimators=cvresult.shape[0])
    # fit
    print('模型拟合中...')
    model.fit(X_train,
              y_train,
              eval_metric='auc')


def get_tuned_xgb(target):
    if target == 'toxic':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=500,
            # max_depth=3,
            # min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=12,
            seed=36)
    elif target == 'severe_toxic':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=109,
            # max_depth=3,
            # min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=12,
            seed=36)
    elif target == 'obscene':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=248,
            # max_depth=3,
            # min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=12,
            seed=36)
    elif target == 'threat':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=146,
            # max_depth=3,
            # min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=12,
            seed=36)
    elif target == 'insult':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=433,
            # max_depth=3,
            # min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=12,
            seed=36)
    elif target == 'identity_hate':
        return XGBClassifier(
            learning_rate=0.2,  # eta
            n_estimators=172,
            # max_depth=3,
            # min_child_weight=1,
            # gamma=0.0,
            # subsample=0.8,
            # colsample_bytree=0.7,
            # reg_alpha=1e-08,        # alpha
            # reg_lambda=1,       # lambda
            # scale_pos_weight=1,
            objective='binary:logistic',
            nthread=4,
            seed=36)

feat_importaces = pd.DataFrame()


xgb_model = get_tuned_xgb(target)

# =============== single model ===============
# print("train: ", target)
# modelfit(xgb_model,
#          X_train=train_features,
#          y_train=train_data[target],
#          useTrainCV=False)
# submission[target] = xgb_model.predict_proba(test_features)[:, 1]
#
# feat_importaces[target] = pd.Series(xgb_model.feature_importances_)

# =================格点搜索================
print("tuning: ", target)
# 格点搜索参数
param_test = {
   'scale_pos_weight': [0.1, 0.5, 1.0]
}

gsearch = GridSearchCV(
  estimator=xgb_model,
  param_grid=param_test,
  scoring='roc_auc',
  iid=False,
  cv=2,
  n_jobs=3,
)

# print('正在搜索...')
gsearch.fit(train_features, train_data[target])

# 输出搜索结果
print('\n当前格点搜索的参数：\n', param_test)
print('\ngsearch.best_params_:', gsearch.best_params_,)
print('\ngsearch.best_score_:', gsearch.best_score_, )

# submission.to_csv('../../output/v1/xgb_submission.csv', index=False)
# feat_importaces.to_csv('../../output/v1/xgb_feat_imp.csv', index=False)

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
