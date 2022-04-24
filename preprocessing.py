import pandas as pd
import numpy as np
import collections
import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
punctuation += '«»1234567890'
from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer("russian")
stop_words = stopwords.words('russian')
stop_words.remove('не')

def preprocessing_text(text: str):
    tokens = word_tokenize(text.lower(), language='russian')
    tokens = [i for i in tokens if i not in punctuation and i not in stop_words]
    tokens = [stemmer.stem(i) for i in tokens]
    return tokens

def get_corpus(x, fake):    
    data = pd.DataFrame(x['title'][x['is_fake'] == fake])
    data.dropna()

    corpus = ''
    for i in data['title']:
        corpus += i

    preprocessed_corpus = preprocessing_text(corpus)
    list_of_words = []
    for i in preprocessed_corpus:
        list_of_words.append(str(i))
    return list_of_words

def freq_words(x):
    frequency = collections.Counter(x)
    most_freq = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:20])
    freq_df = pd.DataFrame({'word': list(most_freq.keys()), 'count': list(most_freq.values())})
    plt.figure(figsize=(20,5))
    sns.barplot(x='word', y='count', data=freq_df)