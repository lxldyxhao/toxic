import numpy as np
import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.sparse import load_npz
from sklearn.feature_selection import SelectFromModel

import time

time_begin = time.time()

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')
test_feats = pd.read_csv('../../output/v1/test_feats.csv').fillna(' ')

train_features = load_npz("../../output/v1_train/lr_train_feats.npz")
test_features = load_npz("../../output/v1_train/lr_test_feats.npz")
print("Read file finished.")

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
#     stop_words='english',
#     ngram_range=(1, 1),
#     max_features=10000)
# word_vectorizer.fit(all_clean_text)
# print("Word TfidfVectorizer fit finished.")
#
# train_word_features = word_vectorizer.transform(train_clean_text)
# test_word_features = word_vectorizer.transform(test_clean_text)
# print("Word TfidfVectorizer transform finished.")
#
# char_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='char',
#     stop_words='english',
#     ngram_range=(2, 6),
#     max_features=50000)
# char_vectorizer.fit(all_text)
# print("Char TfidfVectorizer fit finished.")
#
# train_char_features = char_vectorizer.transform(train_text)
# test_char_features = char_vectorizer.transform(test_text)
# print("Char TfidfVectorizer tranform finished.")
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
#
# del char_vectorizer
# del word_vectorizer
# del train_char_features
# del train_word_features
# del test_char_features
# del test_word_features
# gc.collect()

scores = []
submission = pd.DataFrame.from_dict({'id': test_feats['id']})
for class_name in class_names:
    train_target = train_data[class_name]
    print("Train: ", class_name)
    classifier = LogisticRegression(solver='sag')
    sfm = SelectFromModel(classifier, threshold=0.2)
    print("Before feature selection: ", train_features.shape)
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    print("After feature selection: ", train_sparse_matrix.shape)
    test_sparse_matrix = sfm.transform(test_features)

    classifier = LogisticRegression(max_iter=100,
                                    penalty='l1',
                                    C='1',
                                    class_weight='balanced',
                                    solver='sag',
                                    )
    cv_score = np.mean(cross_val_score(classifier,
                                       train_sparse_matrix,
                                       train_target,
                                       cv=3,
                                       scoring='roc_auc',
                                       n_jobs=-1))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_sparse_matrix, train_target)

    submission[class_name] = classifier.predict_proba(test_sparse_matrix)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('../../output/v1/lr_submission.csv', index=False)
print("File save finised.")

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
