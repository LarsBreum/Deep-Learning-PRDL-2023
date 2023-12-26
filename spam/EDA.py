from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection as skm
from PIL import Image
import matplotlib.pyplot as plt
from nltk.util import ngrams
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize


nltk.download('stopwords')

df = pd.read_csv('/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/spam-data-clean.csv')

print("All:\n",df.head(),"\n")

spam = df[df['Label'] == 'spam']
ham = df[df['Label'] == 'ham']

print("spam:\n",spam.describe(), "\n")
print("ham:\n",ham.describe())


def extract_ngrams(text, n):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        n_grams = list(ngrams(tokens, n))
        return [' '.join(gram) for gram in n_grams]
    else:
        return []

def extract_ngrams_from_dataframe(dataframe, n):
    dataframe['Ngrams'] = dataframe['Text'].apply(lambda x: extract_ngrams(x, n))
    return dataframe

spam_2_gram = extract_ngrams_from_dataframe(spam, 2)
ham_2_gram = extract_ngrams_from_dataframe(ham, 2)

spam_flat = [item for sublist in spam_2_gram['Ngrams'].tolist() for item in sublist]
ham_flat = [item for sublist in ham_2_gram['Ngrams'].tolist() for item in sublist]

spam_2_gram_count = pd.Series(spam_flat).value_counts()
ham_2_gram_count = pd.Series(ham_flat).value_counts()


top_ngrams = spam_2_gram_count.head(20)
top_ngrams.plot(kind='barh', color="red")
plt.xlabel(f'{2}-grams spam')
plt.ylabel('Frequency')
plt.title(f'Top 20 {2}-grams spam')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

plt.show()

top_ngrams = ham_2_gram_count.head(20)
top_ngrams.plot(kind='barh', color='blue')
plt.xlabel(f'{2}-grams ham')
plt.ylabel('Frequency')
plt.title(f'Top 20 {2}-grams ham')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

plt.show()

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


fig, ax = plt.subplots(figsize = (10, 5))


plt.barh(list(spam_word_count["Values"].head(20)),
         list(spam_word_count["count"].head(20)), 
            color ='red')

fig, ax = plt.subplots(figsize = (10, 5))

plt.barh(list(ham_word_count["Values"].head(20)),
         list(ham_word_count["count"].head(20)), 
            color ='blue')

plt.show()