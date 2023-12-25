from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection as skm
from PIL import Image
import matplotlib.pyplot as plt
from nltk.util import ngrams
import nltk
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('stopwords')

df = pd.read_csv('/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/spam-data-clean.csv')

print("All:\n",df.head(),"\n")

spam = df[df['Label'] == 'spam']
ham = df[df['Label'] == 'ham']

print("spam:\n",spam.describe(), "\n")
print("ham:\n",ham.describe())

def ngramconvert(df,n=2):
    for item in df.columns:
        df['new'+item]=df[item].apply(lambda sentence: list(ngrams(sentence.split(), n)))
    return df


# spam_2_gram = ngramconvert(spam,2)
# ham_2_gram = ngramconvert(ham, 2)


spam_word_count = (spam['Text'].str.split(expand=True)
                   .stack()
                   .value_counts()
                   .rename_axis("Values")
                   .reset_index(name='count'))

ham_word_count = (ham['Text'].str.split(expand=True)
                   .stack()
                   .value_counts()
                   .rename_axis("Values")
                   .reset_index(name='count'))

print("spam count: \n",spam_word_count.head(20))
print("ham count: \n",ham_word_count.head(20))


fig, ax = plt.subplots(figsize = (10, 5))


plt.barh(list(spam_word_count["Values"].head(20)),
         list(spam_word_count["count"].head(20)), 
            color ='red')

fig, ax = plt.subplots(figsize = (10, 5))

plt.barh(list(ham_word_count["Values"].head(20)),
         list(ham_word_count["count"].head(20)), 
            color ='blue')

plt.show()