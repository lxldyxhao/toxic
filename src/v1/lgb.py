import gc
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

import time
from scipy.sparse import load_npz

time_begin = time.time()

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')
test_feats = pd.read_csv('../../output/v1/test_feats.csv').fillna(' ')
print('Loaded')

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

# ================= add features ==================

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

# train_data.drop('comment_text', axis=1, inplace=True)
# del test_feats
# del train_text
# del test_text
# del train_char_features
# del test_char_features
# del train_word_features
# del test_word_features
# del char_vectorizer
# del word_vectorizer
# gc.collect()

submission = pd.DataFrame.from_dict({'id': test_feats['id']})

train_features = load_npz("../../output/v1_train/xgb_train_feats.npz")
test_features = load_npz("../../output/v1_train/xgb_test_feats.npz")

for class_name in class_names:
    print(class_name)
    train_target = train_data[class_name]
    # model = LogisticRegression(solver='sag')
    # sfm = SelectFromModel(model, threshold=0.2)
    # print("Before feature selection: ", train_features.shape)
    # train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    # print("After feature selection: ", train_sparse_matrix.shape)
    # test_sparse_matrix = sfm.transform(test_features)

    train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = \
        train_test_split(train_features, train_target, test_size=0.05, random_state=144)

    d_train = lgb.Dataset(train_sparse_matrix, label=y_train)
    d_valid = lgb.Dataset(valid_sparse_matrix, label=y_valid)
    watchlist = [d_train, d_valid]

    params = {'learning_rate': 0.2,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 2,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.6,
              'nthread': 4,
              'lambda_l1': 1,
              'lambda_l2': 1,
              'device': "gpu"}
    rounds_lookup = {'toxic': 140,
                     'severe_toxic': 50,
                     'obscene': 80,
                     'threat': 80,
                     'insult': 70,
                     'identity_hate': 80}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[class_name],
                      valid_sets=watchlist,
                      verbose_eval=10)
    submission[class_name] = model.predict(test_features)

submission.to_csv('../../output/v1/lgb_submission.csv', index=False)

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
