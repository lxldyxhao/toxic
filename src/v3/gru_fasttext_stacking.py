import numpy as np

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import time

time_begin = time.time()

import warnings

warnings.filterwarnings('ignore')

import os
import getopt, sys

opts, args = getopt.getopt(sys.argv[1:], "t:")
now = 0
for (i, j) in opts:
    if i == '-t':
        now = int(j)
print(now)

os.environ['OMP_NUM_THREADS'] = '4'

EMBEDDING_FILE = '../../external_data/crawl-300d-2M.vec'

train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')
submission = pd.read_csv('../../input/sample_submission.csv')
print("Read file finished.")

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
print("Read vec file finished.")

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
print("Embedding finished.")


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
print("Get model finished.")

batch_size = 32
epochs = 2

# X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
# RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

# todo split xtrain
kf = KFold(n_splits=5, shuffle=True, random_state=64)
kf_x_test = []
test_index_needed = []
X_tra = []
y_tra = []

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    if now == i:
        X_tra = x_train[train_index]
        y_tra = y_train[train_index]
        test_index_needed = test_index
        kf_x_test = x_train[test_index_needed]

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=2)
print("Fit model finished.")

stack_train = pd.DataFrame(train['id'][:])
cols = ["gru_stacking_train_toxic", "gru_stacking_train_severe_toxic",
        "gru_stacking_train_obscene", "gru_stacking_train_threat",
        "gru_stacking_train_insult", "gru_stacking_train_identity_hate"]
for i in cols:
    stack_train[i] = 0
stack_train.iloc[test_index_needed, 1:] = model.predict(kf_x_test)
stack_train.to_csv('../../output/v3/gru_stacking_train_' + str(now) + '.csv', index=False)

y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.columns = ["id", "gru_stacking_test_toxic", "gru_stacking_test_severe_toxic",
                      "gru_stacking_test_obscene", "gru_stacking_test_threat",
                      "gru_stacking_test_insult", "gru_stacking_test_identity_hate"]
submission.to_csv('../../output/v3/gru_stacking_test_' + str(now) + '.csv', index=False)

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
