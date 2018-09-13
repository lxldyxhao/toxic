import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

import time
from scipy.sparse import load_npz

time_begin = time.time()

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_data = pd.read_csv('../../output/v1/train_data.csv').fillna(' ')
test_feats = pd.read_csv('../../output/v1/test_feats.csv').fillna(' ')


train_features = load_npz("../../output/v1_train/lr_train_feats.npz")
test_features = load_npz("../../output/v1_train/lr_test_feats.npz")

print("Read file finished.")

scores = []
submission = pd.DataFrame.from_dict({'id': test_feats['id']})
for class_name in class_names:
    train_target = train_data[class_name]
    classifier = LogisticRegression(max_iter=300,
                                    penalty='l2',
                                    C=1,
                                    class_weight='balanced',
                                    solver='sag',)
    cv_score = np.mean(cross_val_score(classifier,
                                       train_features,
                                       train_target,
                                       cv=3,
                                       scoring='roc_auc',
                                       n_jobs=-1))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('../../output/v1/lr_submission.csv', index=False)
print("File save finised.")

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
