# import required packages
import pandas as pd
import numpy as np
import time

# settings
time_begin = time.time()

# importing the dataset
train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')
df = pd.concat([train.iloc[:, 0:2], test.iloc[:, 0:2]]).reset_index(drop=True)
print("Read file finished.")

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

new_train = train[:]

for i in range(0, 3):
    positive_samples = train[train[class_names].sum(axis=1) > 0][:].reset_index()
    positive_samples['rand'] = np.random.random(positive_samples.shape[0])
    shuffled = positive_samples.sort_values(by='rand').reset_index(drop=True)

    positive_samples['comment_text'] = positive_samples['comment_text'] + shuffled['comment_text']
    positive_samples['toxic'] = (positive_samples['toxic'] + shuffled['toxic']).apply(lambda x: 1 if x > 1 else x)
    positive_samples['severe_toxic'] = (positive_samples['severe_toxic'] + shuffled['severe_toxic']).apply(
        lambda x: 1 if x > 1 else x)
    positive_samples['obscene'] = (positive_samples['obscene'] + shuffled['obscene']).apply(lambda x: 1 if x > 1 else x)
    positive_samples['threat'] = (positive_samples['threat'] + shuffled['threat']).apply(lambda x: 1 if x > 1 else x)
    positive_samples['insult'] = (positive_samples['insult'] + shuffled['insult']).apply(lambda x: 1 if x > 1 else x)
    positive_samples['identity_hate'] = (positive_samples['identity_hate'] + shuffled['identity_hate']).apply(
        lambda x: 1 if x > 1 else x)
    positive_samples['id'] = positive_samples['id'] + "+" + shuffled['id']

    positive_samples = positive_samples.drop(['index', 'rand'], axis=1)
    new_train = pd.concat([new_train, positive_samples])
print("Data increase finished.")

new_train = new_train.sample(frac=1).reset_index(drop=True)
new_train.to_csv('../../output/v3/increased_train.csv', index=False)
test.to_csv('../../output/v3/test.csv', index=False)
print("Save data finished.")

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))
