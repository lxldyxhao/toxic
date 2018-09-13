# import required packages
import pandas as pd
import numpy as np
from collections import Counter
import string
import re  # for regex
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn import preprocessing
import time
import warnings

# settings
time_begin = time.time()
eng_stopwords = set(stopwords.words("english"))

# importing the dataset
train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')
merge = pd.concat([train.iloc[:, 0:2], test.iloc[:, 0:2]])
df = merge.reset_index(drop=True)

print("Read file finished.")

# ===============Indirect features====================

df['splited_str'] = df["comment_text"].apply(lambda x: str(x).split())

df['count_sent'] = df["comment_text"].apply(lambda x: len(re.findall("\n", str(x))) + 1)
df['count_word'] = df["splited_str"].apply(lambda x: len(x))
df['count_unique_word'] = df["splited_str"].apply(lambda x: len(set(x)))
df['count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))
df["count_punctuations"] = df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df["count_words_upper"] = df["splited_str"].apply(lambda x: len([w for w in x if w.isupper()]))
df["count_words_title"] = df["splited_str"].apply(lambda x: len([w for w in x if w.istitle()]))
df["count_stopwords"] = df["comment_text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
df["count_!"] = df["comment_text"].apply(lambda x: len([c for c in str(x) if c == '!']))
df["count_?"] = df["comment_text"].apply(lambda x: len([c for c in str(x) if c == '?']))
df["count_duplicate"] = df['count_word'] - df['count_unique_word']
df["count_space"] = df["comment_text"].apply(lambda x: len([c for c in str(x) if c == ' ' or c == '\t']))
df["count_number"] = df["comment_text"].apply(lambda x: len(re.findall("[1-9]\d*", str(x))))


def get_max_upper_len(li):
    up_len = [len(w) for w in li if w.isupper()]
    return 0 if len(up_len) == 0 else max(up_len)


def get_max_word_len(li):
    word_len = [len(re.sub('[a-zA-z]+://[^\s]*', "", w)) for w in li]
    return 0 if len(word_len) == 0 else max(word_len)


def get_max_number_len(s):
    nums = re.findall("[1-9]\d*", str(s))
    num_lens = [len(i) for i in nums]
    return 0 if len(num_lens) == 0 else max(num_lens)


df["max_duplicate_time"] = df["splited_str"].apply(lambda x: max(Counter(x).values()) if len(x) != 0 else 0)
df["max_upper_len"] = df["splited_str"].apply(get_max_upper_len)
df["max_word_len"] = df["splited_str"].apply(get_max_word_len)  # should not include links
df["max_number_len"] = df["comment_text"].apply(get_max_number_len)

df["percent_max_duplicate"] = df["max_duplicate_time"] * 100 / df['count_word']
df["percent_upper"] = df["count_words_upper"] * 100 / df['count_word']
df["percent_title"] = df["count_words_title"] * 100 / df['count_word']
df["percent_!"] = df["count_!"] * 100 / df['count_letters']
df["percent_?"] = df["count_?"] * 100 / df['count_letters']
df["percent_number"] = df["count_number"] * 100 / df['count_word']
df['percent_word_unique'] = df['count_unique_word'] * 100 / df['count_word']
df['percent_punct'] = df['count_punctuations'] * 100 / df['count_letters']


def get_mean_or_zero(li):
    return 0 if len(li) == 0 else np.mean(li)


df["mean_word_len"] = df["splited_str"].apply(lambda x: get_mean_or_zero([len(w) for w in x]))
df["mean_unique_word"] = df["splited_str"].apply(lambda x: get_mean_or_zero([len(i) for i in list(set(x))]))
df["mean_upper"] = df["splited_str"].apply(lambda x: get_mean_or_zero([len(w) for w in x if w.isupper()]))
df["mean_title"] = df["splited_str"].apply(lambda x: get_mean_or_zero([len(w) for w in x if w.istitle()]))
df["mean_number"] = df["comment_text"].apply(
    lambda x: get_mean_or_zero([len(w) for w in re.findall("[1-9]\d*", str(x))]))

df = df.drop(labels='splited_str', axis=1).fillna(0)

print("Indirect features finished.")

# ==================Leaky features=========================

df['ip'] = df["comment_text"].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", str(x)))
df['count_ip'] = df["ip"].apply(lambda x: len(x))
df['link'] = df["comment_text"].apply(lambda x: re.findall("[a-zA-z]+://[^\s]*", str(x)))
df['count_links'] = df["link"].apply(lambda x: len(x))
df['article_id'] = df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$", str(x)))
df['article_id_flag'] = df["article_id"].apply(lambda x: len(x))
df['username'] = df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|", str(x)))
df['count_usernames'] = df["username"].apply(lambda x: len(x))

print("Leakage features finished.")

# ==================corpus cleaning======================

# https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'd": "I had",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    "didn't": "did not",
    "tryin'": "trying"
}

corpus = merge.comment_text
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()


def clean(comment):
    comment = comment.lower()
    comment = re.sub("\\n", "", comment)
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    comment = re.sub("\[\[.*\]", "", comment)

    words = tokenizer.tokenize(comment)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]

    clean_sent = " ".join(words)
    return (clean_sent)


def soft_clean(comment):
    comment = comment.lower()
    comment = re.sub("\\n", "", comment)
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    words = tokenizer.tokenize(comment)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]

    # remove words longer than 30 chars
    result = []
    for word in words:
        if word != None and len(word) <= 30:
            result.append(word)

    # remove punctuations
    clean_sent = " ".join(result)
    for i in string.punctuation + '‘’':
        if i not in ',.!?':
            clean_sent = clean_sent.replace(i, '')

    return (clean_sent)


df['clean_text'] = df['comment_text'].apply(lambda x: clean(str(x)))
df['soft_clean_text'] = df['comment_text'].apply(lambda x: soft_clean(str(x)))
print("Corpus cleaning finished.")

# =================== standration =====================

COLS = ['id', 'comment_text', 'clean_text','soft_clean_text',
        'count_sent', 'count_word', 'count_unique_word',
        'count_letters', 'count_punctuations', 'count_words_upper',
        'count_words_title', 'count_stopwords', 'count_!', 'count_?',
        'count_duplicate', 'count_space', 'count_number', 'max_duplicate_time',
        'max_upper_len', 'max_word_len', 'max_number_len',
        'percent_max_duplicate', 'percent_upper', 'percent_title', 'percent_!',
        'percent_?', 'percent_number', 'percent_word_unique', 'percent_punct',
        'mean_word_len', 'mean_unique_word', 'mean_upper', 'mean_title',
        'mean_number', 'ip', 'count_ip', 'link', 'count_links', 'article_id',
        'article_id_flag', 'username', 'count_usernames']

NUM_COLS = ['count_sent', 'count_word', 'count_unique_word',
            'count_letters', 'count_punctuations', 'count_words_upper',
            'count_words_title', 'count_stopwords', 'count_!', 'count_?',
            'count_duplicate', 'count_space', 'count_number', 'max_duplicate_time',
            'max_upper_len', 'max_word_len', 'max_number_len',
            'percent_max_duplicate', 'percent_upper', 'percent_title', 'percent_!',
            'percent_?', 'percent_number', 'percent_word_unique', 'percent_punct',
            'mean_word_len', 'mean_unique_word', 'mean_upper', 'mean_title',
            'mean_number', 'count_ip', 'count_links', 'article_id_flag', 'count_usernames']

TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

logged_num = df[NUM_COLS].apply(lambda x: np.log(x + 1))
minmax_scaler = preprocessing.MinMaxScaler()
X_scaled = minmax_scaler.fit_transform(logged_num)
X_scaled = pd.DataFrame(X_scaled, columns=NUM_COLS)

for col in X_scaled.columns:
    df[col] = X_scaled[col]

# ====================final data======================

# serperate df into train and test features
# df does not include ngrams
train_feats = df.iloc[0:len(train), :]
train_tags = train.iloc[:, 2:]
train_data = pd.concat([train_feats, train_tags], axis=1)
test_feats = df.iloc[len(train):, ].reset_index(drop=True)

# df.to_csv('/home/disk_data/Toxic-kaggle/output/v0/all_feats.csv', index=True)
train_data.to_csv('../../output/v1/train_data.csv', index=False)
test_feats.to_csv('../../output/v1/test_feats.csv', index=False)
print("Save data finished.")

time_spend = time.time() - time_begin
print('\n运行时间：%d 秒，约%d分钟\n' % (time_spend, time_spend // 60))

# import matplotlib.pyplot
# import seaborn as sns
# def draw(curr):
#     plt.title(curr)
#     ax = sns.kdeplot(df[df.index < 159571][curr], label="Train", shade=True, color='r')
#     ax = sns.kdeplot(df[df.index >= 159571][curr], label="Test")
