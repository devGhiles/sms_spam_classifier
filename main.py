import sys
import pickle
import numpy as np


# read the text message
try:
    filepath = sys.argv[1]
except IndexError:
    print('No text file given!')
    exit()

try:
    with open(filepath, 'r') as f:
        text = f.read()
except FileNotFoundError:
    print('File {0} not found!'.format(filepath))
    exit()


# load the vectorizer and the classifier
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('clf.pkl', 'rb') as f:
        clf = pickle.load(f)
except FileNotFoundError as e:
    print('File {0} not found! Make sure to run "training.py" in the same directory first.'.format(e.filename))
    exit()


# feature extraction from the text
X = np.array(vectorizer.transform([text]).todense())


# label prediction
y = clf.predict(X)[0]
probas = clf.predict_proba(X)[0]


# print the result
print('Message: {0}'.format(text))
print('Category: ', end='')
if y == 0:
    print('Not spam')
else:
    print('Spam')
print('Confidence level: {0}%'.format(round(100 * probas[y], 2)))