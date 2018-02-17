import pickle
import nltk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# read the data
data = pd.read_csv('spam.csv', encoding='latin-1')


# data preprocessing
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
data['label'] = data['label'].apply(lambda label: 1 if label == 'spam' else 0)


# feature extraction
tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.tokenize.word_tokenize, stop_words='english', lowercase=True)
X = tfidf_vectorizer.fit_transform(data['message'])
X = np.array(X.todense())
y = data['label'].values


# create and train the model; details on how the model was chosen are in the README.md file
clf = MultinomialNB(alpha=0.1)
clf.fit(X, y)


# save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
