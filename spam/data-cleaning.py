from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import pandas as pd
import sklearn.model_selection as skm
import glob
import string
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('stopwords')

nltk.download('punkt')

def preprocess_text(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    text = remove_hex_numbers(text)
    text = remove_numbers(text)
    text = text.strip()
    return text

def perform_stemming(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_email_headers(email_text):
    header_pattern = r'^.*?\n\n'
    email_body = re.sub(header_pattern, '', email_text, flags=re.DOTALL)
    return email_body

def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    text_without_html = soup.get_text(separator=' ')
    return text_without_html

# Function to remove numbers using regular expressions
def remove_numbers(text):
    text_without_numbers = re.sub(r'\d+', '', text)
    return text_without_numbers

# Function to remove hexadecimal numbers using regular expressions
def remove_hex_numbers(text):
    text_without_hex = re.sub(r'\b[0-9a-fA-F]+\b', '', text)
    return text_without_hex

def remove_hyperlinks(text):
    text_without_links = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text_without_links

def remove_non_utf8_characters(text):
    try:
        # Try to encode the text as UTF-8 and decode it back
        text = text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        # If decoding as UTF-8 fails, replace non-UTF-8 characters with an empty string
        text = ''.join(char if ord(char) < 128 else '' for char in text)
    return text


def read_file(path, label):
    file_list = glob.glob(path)

    df = pd.DataFrame()
    for file_path in file_list:
        with open(file_path, 'r', encoding="latin-1") as file:
            file_content = file.read()
            content = remove_non_utf8_characters(file_content)

        df = df.append({'Text': content, 'Label': label}, ignore_index=True)
    
    return df


easy_ham_path = '/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/easy_ham/*.*'
easy_hame_2_path = '/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/easy_ham_2/*.*'
hard_ham_path = '/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/hard_ham/*.*'
spam_path = '/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/spam/*.*'
spam_2_path = '/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/spam_2/*.*'

df_easy_ham = read_file(easy_ham_path, "ham")
df_easy_ham_2 = read_file(easy_hame_2_path, "ham")
df_hard_ham = read_file(hard_ham_path, "ham")
df_spam = read_file(spam_path, "spam")
df_spam_2 = read_file(spam_2_path, "spam")


df = pd.concat([df_easy_ham, df_easy_ham_2, df_hard_ham, df_spam, df_spam_2])

df['Text'] = df['Text'].apply(remove_email_headers).apply(remove_hyperlinks).apply(remove_html_tags).apply(preprocess_text).apply(remove_stop_words).apply(perform_stemming).apply(perform_lemmatization).apply(remove_non_utf8_characters)

df_clean = df.dropna()

df_clean.to_csv('/home/larsbreum/code/PRDL/Project-Deep-Learning/spam/data/spam-data-clean.csv', index=False) 

print(df_clean.describe())
