# -*- coding=utf-8 -*-
import gc
import time
import pandas as pd
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBClassifier
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import load_npz, save_npz
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from sklearn.feature_extraction.text import TfidfVectorizer

time_begin = time.time()

# ---------------读取数据---------------------

train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')
test_feats = pd.read_csv('../../output/v1/test_feats.csv').fillna(' ')

# # =================== tfidf ======================
#
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
# word_vectorizer.fit(all_text)
# train_word_features = word_vectorizer.transform(train_text)
# test_word_features = word_vectorizer.transform(test_text)
# print('Word TFIDF finished.')
#
# char_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='char',
#     stop_words='english',
#     ngram_range=(2, 6),
#     max_features=50000)
# char_vectorizer.fit(all_clean_text)
# train_char_features = char_vectorizer.transform(train_clean_text)
# test_char_features = char_vectorizer.transform(test_clean_text)
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
# save_npz("../../output/v3/xgb_exchanged_train_feats.npz", train_features)
# save_npz("../../output/v3/xgb_exchanged_test_feats.npz", test_features)


train_features = load_npz("../../output/v3/xgb_exchanged_train_feats.npz")
test_features = load_npz("../../output/v3/xgb_exchanged_test_feats.npz")
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for class_name in class_names:
    train_target = train_data[class_name]
    print("Train: ", class_name)
    classifier = LogisticRegression(solver='sag')
    sfm = SelectFromModel(classifier, threshold=0.2)
    print("Before feature selection: ", train_features.shape)
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    print("After feature selection: ", train_sparse_matrix.shape)
    test_sparse_matrix = sfm.transform(test_features)

    save_npz('../../output/v3/xgb_exchanged_train_less_' + class_name + '.csv', train_sparse_matrix)
    save_npz('../../output/v3/xgb_exchanged_test_less_' + class_name + '.csv', test_sparse_matrix)
