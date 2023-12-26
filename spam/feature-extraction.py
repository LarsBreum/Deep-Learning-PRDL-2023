from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection as skm
from PIL import Image
import matplotlib.pyplot as plt
from nltk.util import ngrams
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv('/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/spam-data-clean.csv')
df = df.dropna()

texts = df["Text"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts.to_list())

y = df["Label"]

# Naive Bayes
# Splitting the data 50/50 train/test
(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.5, random_state=1)
gnb = GaussianNB()

y_pred = gnb.fit(X_train.toarray(), y_train).predict(X_test.toarray())

print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))


