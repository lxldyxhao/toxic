#import required packages
#basics
import pandas as pd 
import numpy as np

#misc
import gc
import time
import warnings

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

# #viz
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec 
# import seaborn as sns
# from wordcloud import WordCloud ,STOPWORDS
# from PIL import Image
# import matplotlib_venn as venn
# color = sns.color_palette()
# sns.set_style("dark")

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack


#settings
start_time=time.time()
eng_stopwords = set(stopwords.words("english"))

#importing the dataset
train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')

# rowsums=train.iloc[:,2:].sum(axis=1)
# train['clean']=(rowsums==0)

merge = pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
df = merge.reset_index(drop=True)

print("Read file finished.")

#===============Indirect features====================

#Sentense count in each comment: 
    #  '\n' can be used to count the number of sentences in each comment
df['count_sent'] = df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
#Word count in each comment:
df['count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))
#Unique word count
df['count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))
#Letter count
df['count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))
#punctuation count
df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
#Average length of the words
df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

#derived features
#Word count percent in each comment:
df['word_unique_percent'] = df['count_unique_word']*100/df['count_word']
#derived features
#Punct percent in each comment:
df['punct_percent'] = df['count_punctuations']*100/df['count_word']


# train_feats['count_word'].loc[train_feats['count_word']>200] = 200
# train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200
#prep for split violin plots
#For the desired plots , the data must be in long format
# temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')
#spammers - comments with less than 40% unique words
# spammers=train_feats[train_feats['word_unique_percent']<30] 

# todo how to add spam features?


print("Indirect features finished.")

#==================Leaky features=========================
df['ip']=df["comment_text"].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",str(x)))
#count of ip addresses
df['count_ip']=df["ip"].apply(lambda x: len(x))

#links
df['link']=df["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
#count of links
df['count_links']=df["link"].apply(lambda x: len(x))
 
#article ids
df['article_id']=df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))
df['article_id_flag']=df["article_id"].apply(lambda x: len(x))

#username
##              regex for     Match anything with [[User: ---------- ]]
# regexp = re.compile("\[\[User:(.*)\|")
df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))
#count of username mentions
df['count_usernames']=df["username"].apply(lambda x: len(x))
#check if features are created
#df.username[df.count_usernames>0]

print("Leakage features finished.")
 
# ==================corpus cleaning======================

#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

corpus=merge.comment_text

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    #remove \n
    comment=re.sub("\\n","",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)


df['clean_text'] = df['comment_text'].apply(lambda x :clean(str(x)))

print("Corpus cleaning finished.")


# ====================final data======================
COLS = ['id', 'comment_text', 'clean_text', 'count_sent', 'count_word', 'count_unique_word',
       'count_letters', 'count_punctuations', 'count_words_upper',
       'count_words_title', 'count_stopwords', 'mean_word_len',
       'word_unique_percent', 'punct_percent', 'ip', 'count_ip', 'link',
       'count_links', 'article_id', 'article_id_flag', 'username', 'count_usernames']
df = df[COLS]

# serperate df into train and test features
# df does not include ngrams
train_feats = df.iloc[0:len(train),]
train_tags = train.iloc[:,2:] #?????????????????????
train_data = pd.concat([train_feats,train_tags],axis=1)

test_feats = df.iloc[len(train):,].reset_index(drop=True)

#df.to_csv('/home/disk_data/Toxic-kaggle/output/v0/all_feats.csv', index=True)
train_data.to_csv('../../output/v0/train_data.csv', index=True)
test_feats.to_csv('../../output/v0/test_feats.csv', index=True)
print("Save data finished.")

del df
del train_data
del test_feats


# =========================================================
# =========================================================

train_data = pd.read_csv('../../output/v0/train_data.csv')
test_feats = pd.read_csv('../../output/v0/test_feats.csv')
merge = pd.concat([train_data.iloc[:,0:-6],test_feats])
df = merge.reset_index(drop=True)
df['clean_text'] = df['clean_text'].fillna("")
print("Read file finished.")

# ====================direct features==================


### Unigrams -- TF-IDF 
# # using settings recommended here for TF-IDF -- https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

# #some detailed description of the parameters
# # min_df=10 --- ignore terms that appear lesser than 10 times 
# # max_features=None  --- Create as many words as present in the text corpus
#     # changing max_features to 10k for memmory issues
# # analyzer='word'  --- Create features from words (alternatively char can also be used)
# # ngram_range=(1,1)  --- Use only one word at a time (unigrams)
# # strip_accents='unicode' -- removes accents
# # use_idf=1,smooth_idf=1 --- enable IDF
# # sublinear_tf=1   --- Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)


#temp settings to min=200 to facilitate top features section to run in kernals
#change back to min=10 to get better results
clean_corpus = df.clean_text

# word unigram
tfv = TfidfVectorizer(	min_df=10,  
						max_features=30000,
						strip_accents='unicode', 
						analyzer='word',
						ngram_range=(1,1),
						use_idf=1,
						smooth_idf=1,
						sublinear_tf=1,
						stop_words = 'english')
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_unigrams =  tfv.transform(clean_corpus.iloc[:train_data.shape[0]])
test_unigrams = tfv.transform(clean_corpus.iloc[train_data.shape[0]:])

# word bigram
#temp settings to min=150 to facilitate top features section to run in kernals
#change back to min=10 to get better results
tfv = TfidfVectorizer(	min_df=10,  
						max_features=30000, 
						strip_accents='unicode', 
						analyzer='word',
						ngram_range=(2,2),
						use_idf=1,
						smooth_idf=1,
						sublinear_tf=1,
						stop_words = 'english')
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_bigrams =  tfv.transform(clean_corpus.iloc[:train_data.shape[0]])
test_bigrams = tfv.transform(clean_corpus.iloc[train_data.shape[0]:])

# char ngrams
tfv = TfidfVectorizer(	min_df=10,  
						max_features=30000, 
						strip_accents='unicode', 
						analyzer='char',
						ngram_range=(1,4),
						use_idf=1,
						smooth_idf=1,
						sublinear_tf=1,
						stop_words = 'english')
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_charngrams =  tfv.transform(clean_corpus.iloc[:train_data.shape[0]])
test_charngrams = tfv.transform(clean_corpus.iloc[train_data.shape[0]:])

print("Direct Features Finished.")

# =====================set train columns======================
SELECTED_COLS=['count_sent', 'count_word', 'count_unique_word',
       'count_letters', 'count_punctuations', 'count_words_upper',
       'count_words_title', 'count_stopwords', 'mean_word_len',
       'word_unique_percent', 'punct_percent']

TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# all features used to train except leakage features
target_x = hstack((train_bigrams,train_charngrams,train_unigrams,train_feats[SELECTED_COLS])).tocsr()
target_y = train_tags[TARGET_COLS]

# all features for test data except leakage features
test_x = hstack((test_bigrams,test_charngrams,test_unigrams,test_feats[SELECTED_COLS])).tocsr()


# ==================baseline model===========================


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x): 
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self
    



model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)
X_train, X_valid, y_train, y_valid = \
	train_test_split(target_x, target_y, test_size=0.33, random_state=2018)
train_loss = []
valid_loss = []
preds_train = np.zeros((X_train.shape[0], y_train.shape[1]))
preds_valid = np.zeros((X_valid.shape[0], y_valid.shape[1]))

for i, j in enumerate(TARGET_COLS):
    print('Class:= '+j)
    model.fit(X_train,y_train[j])
    preds_valid[:,i] = model.predict_proba(X_valid)[:,1]
    preds_train[:,i] = model.predict_proba(X_train)[:,1]
    train_loss_class=log_loss(y_train[j],preds_train[:,i])
    valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))


end_time=time.time()
print("total time till NB base model creation",end_time-start_time)