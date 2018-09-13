# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from keras.layers import Dense, Input, LST M, Bidirectional, Activation, Conv1D
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Any results you write to the current directory are saved as output.
import getopt, sys
opts, args = getopt.getopt(sys.argv[1:], "t:")
now = 0
for (i,j) in opts:
    if i=='-t':
        now = int(j)
print(now)


EMBEDDING_FILE = '../../external_data/glove.840B.300d.txt'
train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')

train["comment_text"].fillna("fillna")
test["comment_text"].fillna("fillna")

X_train = train["comment_text"].str.lower()
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

X_test = test["comment_text"].str.lower()

max_features = 110000
maxlen = 150
embed_size = 300

tok = text.Tokenizer(num_words=max_features, lower=True)
tok.fit_on_texts(list(X_train) + list(X_test))
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tok.word_index
# prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

sequence_input = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.1)(x)
preds = Dense(6, activation="sigmoid")(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
epochs = 4

# todo split xtrain
kf = KFold(n_splits=5, shuffle=True, random_state=36)
kf_x_test = []
test_index_needed = []
for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    if now == i:
        X_tra = x_train[train_index]
        y_tra = y_train[train_index]
        test_index_needed = test_index
        kf_x_test = x_train[test_index_needed]

file_path = "weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]

model.fit(X_tra, y_tra, batch_size=batch_size, epochs=5, callbacks=callbacks_list,
          verbose=1)

model.load_weights(file_path)

# todo predict x_test
stack_train = pd.DataFrame(train['id'][:])
cols = ["lstm_stacking_train_toxic", "lstm_stacking_train_severe_toxic",
        "lstm_stacking_train_obscene", "lstm_stacking_train_threat",
        "lstm_stacking_train_insult", "lstm_stacking_train_identity_hate"]
for i in cols:
    stack_train[i] = 0
stack_train.iloc[test_index_needed, 1:] = model.predict(kf_x_test)
stack_train.to_csv('../../output/v3/lstm_cnn_stacking_train_' + str(now) + '.csv', index=False)

y_pred = model.predict(x_test, batch_size=1024, verbose=1)
submission = pd.read_csv('../../input/sample_submission.csv')

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.columns = ["id","lstm_stacking_test_toxic", "lstm_stacking_test_severe_toxic",
            "lstm_stacking_test_obscene", "lstm_stacking_test_threat",
            "lstm_stacking_test_insult", "lstm_stacking_test_identity_hate"]
submission.to_csv('../../output/v3/lstm_cnn_stacking_test_' + str(now) + '.csv', index=False)
