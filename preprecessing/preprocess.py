import os
import sys
from pathlib import Path
import numpy as np
import random
import pickle
import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS

import textacy as tx
import re
import nltk
from nltk.tokenize import word_tokenize



class Preprocess():
    def __init__ (self, data, vocab, cities):
        self.path_data = data
        self.path_vocab = vocab
        self.path_cities = cities
        return

    def get_path(self, argument):
        if argument == "data":
            return self.path_data
        if argument == "vocab":
            return self.path_vocab
        if argument == "cities":
            return self.path_cities
        else:
            print("Error")
            return

    def load_pickle(self, path):
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def load_data(self):
        data = []
        for subdir, dirs, files in os.walk(self.path_data):
            for file in files:
                if '.txt' in os.path.join(subdir, file) and 'upload_done' not in os.path.join(subdir, file) and "0001.txt" not in os.path.join(subdir, file):
                    f = open((os.path.join(subdir,file)),"r")
                    data.append(f.read().replace('\n', ' '))
                    f.close()
        return data

    def data_pd(self, data):
        data_df = pd.DataFrame(data, columns=['documents'])
        return data_df

    def to_pickle(self, data, name):
        with open(name + '.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def remove_punkt_num_sym(self, data, min_len = 3):
        processed = []
        for doc in data:
            doc = tx.preprocess.replace_currency_symbols(doc, replace_with = " ")
            doc = tx.preprocess.replace_numbers(doc, replace_with=' ')
            doc = doc.replace('"', '').replace('<', ' ').replace('=', ' ').replace('>', ' ').replace('^', ' ').replace('±', ' ').replace('¨', ' ').replace('¤', ' ').replace('´', ' ').replace('°', ' ').replace('■', ' ')
            doc = tx.preprocess.normalize_whitespace(doc)
            doc = tx.preprocess.preprocess_text(doc, lowercase=True, no_punct=True)
            doc = re.sub(r'\b\w{1,3}\b', '', doc)
            processed.append(doc)
        return processed


    def tokenize_nltk(self, data):
        processed = []
        for doc in data:
            doc = tx.preprocess.normalize_whitespace(doc)
            doc = tx.preprocess.preprocess_text(doc, lowercase=True, no_punct=True)
            doc = re.sub(r'\b\w{1,3}\b', '', doc)
            tokens = word_tokenize(doc)
            words = [word for word in tokens if word.isalpha()]
            processed.append(words)
        return processed


    def tokenize(self, data):
        token_list = []
        for doc in data:
            tokens = re.sub("[^\w]", " ", doc).split()
            token_list.append(tokens)
            tokens = ""
        return token_list


    def spacy_stopwords(self, data):
        nlp = spacy.load('de_core_news_sm')
        new = ['gmbh', 'ii', 'bitte', 'ag', 'iban', 'dr.', 'i', 'bic', 'bank',
               'namen', 'überweisung', 'euro', 'angabe', 'gogreen',
               'verwendungszweck', 'betrag', 'bic','werden', 'über', 'können',
               'unter', 'dass', 'bzw', 'monat', 'digitalkasten', 'mail', 'sowie',
               'ergo', 'seite', 'bessemerstr', 'bessemerstraße', 'gogreen', 'klimaneutraler',
               'allianz', 'sehr', 'geehrter', 'grüssen', 'grüßen','erfolgt','daten','dsgvo',
               'post', 'versand', 'deutsche', 'deutsch', 'deutschen', 'berlin', 'zustellen',
               'werde', 'werden' 'weit', 'datum', 'zustellen', 'kosten', 'telefon', 'informationen',
               'email', 'erhalten', 'payback', 'gelten', 'gültig', 'person', 'höhe', 'allgemein','green',
               'gogreen', 'gqgreen', 'dialogpost', 'herr', 'herrn', 'sehr', 'geehrte', 'geehrter'
              ]
        for stopword in new:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True

        filtered_list = []
        for doc in data:
            text = nlp(doc)
            words = [w for w in text if not w.is_stop]
            filtered_list.append(words)
            words = ""
        return filtered_list



    def remove_stopwords(self, data, more_stops=""):
        stopwords = STOP_WORDS

        new = {'gmbh', 'ii', 'bitte', 'ag', 'iban', 'dr.', 'i', 'bic', 'bank',
               'namen', 'überweisung', 'euro', 'angabe', 'gogreen',
               'verwendungszweck', 'betrag', 'bic','werden', 'über', 'können',
               'unter', 'dass', 'bzw', 'monat', 'digitalkasten', 'mail', 'sowie',
               'ergo', 'seite', 'bessemerstr', 'bessemerstraße', 'gogreen', 'klimaneutraler',
               'allianz', 'sehr', 'geehrter', 'grüssen', 'grüßen','erfolgt','daten','dsgvo',
               'post', 'versand', 'deutsche', 'deutsch', 'deutschen', 'berlin', 'zustellen',
               'werde', 'werden' 'weit', 'datum', 'zustellen', 'kosten', 'telefon', 'informationen',
               'email', 'erhalten', 'payback', 'gelten', 'gültig', 'person', 'höhe', 'allgemein','green',
               'gogreen', 'gqgreen', 'dialogpost', 'herr', 'herrn', 'sehr', 'geehrte', 'geehrter'
               }
        stopwords.update(new)
        stopwords.update(more_stops)
        stopwords = set(stopwords)
        filtered_list = []
        for doc in data:
            words = [w for w in doc if not w in stopwords]
            filtered_list.append(words)
            words = ""
        return filtered_list

    def remove_non_vocab(self, data):
        vocab = set(self.load_pickle(self.path_vocab))
        filtered_list = []
        filtered_doc = []
        for doc in data:
            for word in doc:
                if word in vocab:
                    filtered_doc.append(word)
            filtered_list.append(filtered_doc)
            filtered_doc = []
        return filtered_list

    def remove_cities(self, data):
        cities = set(self.load_pickle(self.path_cities))
        filtered_cities = []
        filtered_doc = []
        for doc in data:
            for word in doc:
                if word not in cities:
                    filtered_doc.append(word)
            filtered_cities.append(filtered_doc)
            filtered_doc = []
        return filtered_cities

    def remove_anything(self, data, path_any):
        list = set(self.load_pickle(path_any))
        filtered_data = []
        filtered_doc = []
        for doc in data:
            for word in doc:
                if word not in list:
                    filtered_doc.append(word)
            filtered_data.append(filtered_doc)
            filtered_doc = []
        return filtered_data


class LDA():
    def __init__(self, data):
        self.data = data
        print("What function do you want to write?")

        return




def main():
    path_data = '1826'
    path_vocab = 'vocab.pickle'
    path_cities = 'cities.pickle'
    prep = Preprocess(path_data,path_vocab,path_cities)
    data = prep.load_data()
    data = prep.remove_punkt_num_sym(data)
    #data = prep.spacy_stopwords(data)
    data = prep.tokenize_nltk(data)
    data = prep.remove_stopwords(data)
    data = prep.remove_non_vocab(data)
    data = prep.remove_cities(data)
    prep.to_pickle(data, "processed_new")
    return data



if  __name__ =='__main__':
    main()
