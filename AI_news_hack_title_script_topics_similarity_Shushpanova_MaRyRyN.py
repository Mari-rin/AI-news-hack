#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
from langdetect import detect 
import re
import string
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pymorphy2
from tqdm.auto import tqdm, trange
from nltk import word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.decomposition import NMF
import pickle



def split_data_phrase(text): 
    usr_name = re.compile(" («»@—(\w|\_)*)\W")
    result = usr_name.sub("", str(text)) 
    for punct in string.punctuation:
        if punct in result:
            result = result.replace(punct, '')
    result = result.replace("\n", "")
    return result

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_kav(text): 
    result = text.replace('»', '')
    result = result.replace('«', '')
    result = result.replace('—', '')
    result = re.sub("[0-9]", "", result)
    for asci in string.ascii_letters:
        if asci in result:
            result = result.replace(asci, '')
    result = result.replace("\n", "")
    return result

def clean_text(df, column):
    df[column] = df[column].apply(split_data_phrase)
    df[column] = df[column].apply(remove_emoji)
    df[column] = df[column].apply(remove_kav)
    return df


def pre_text_morph(txt):
    morph = pymorphy2.MorphAnalyzer()
    txt = str(txt)
    txt = [morph.parse(word)[0].normal_form for word in txt.split()]
    return " ".join((map(str, txt)))

def morph_txt(df, column):
    df[column] = df[column].apply(pre_text_morph)
    return df


def null_out(df, column):
    nan_posts_title = df[df[column].notnull() == False]
    posts2_new = df[~ df[column].isin(nan_posts_title[column])]
    return posts2_new

def tfidf_vec(df, column):
    my_stop_words = ['год', 'месяц', 'день', 'час', 'минута', 'секунда', 'век', 'эпоха', 'январь', 'февраль', 'март', 
                 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь', 'это', 'свой', 
                 'день', 'ночь', 'всё',
                 'сегодня', 'завтра']
    politic_list = ['контрнаступление', 'конфликт', 'оружие', 'переговоры', 'боеприпас', 'война',
              'военный', 'кассетный', 'нато', 'вооружённый', 'беспилотник', 'обстрел', 'боец', 'нападение',
              'взрыв', 'мятеж', 'операция', 'санкция']
    stop_words = list(get_stop_words('ru'))
    nltk_stopwords = list(stopwords.words('russian'))

    stop_words.extend(nltk_stopwords)
    stop_words.extend(my_stop_words)
    stop_words.extend(politic_list)
    
    tfidf = TfidfVectorizer(max_df = 0.35, min_df = 2, stop_words = stop_words)
    dtm = tfidf.fit_transform(df[column])
    
    return dtm, tfidf

def nmf_model(tfidf_txt, model = None): 
    if model is not None:
        with open(model, 'rb') as f:
            nmf_model = pickle.load(f)
        nmf_model.fit(tfidf_txt)
    else:
        nmf_model = NMF(n_components = 10, random_state = 42)
        nmf_model.fit(tfidf_txt)
    return nmf_model

def print_top25_words(model, tfidf, n_words = 25):
    for index, topic in enumerate(model.components_):
        print(f'TOP {n_words} words in topic #{index}')
        print([tfidf.get_feature_names()[i] for i in topic.argsort()[-25:]])
        print('\n')
        
def class_topic(model, tfidf_txt, df, column):
    
    classes_dict = {0: 'общее',
               1: 'технологии',
               2: 'политика',
               3: 'шоубиз',
               4: 'образовательный контент',
               5: 'путешествия/релокация',
               6: 'финансы',
               7: 'крипта',
               8: 'fashion',
               9: 'развлечения'}
    
    W = model.transform(tfidf_txt)
    df_w = pd.DataFrame(W)
    df_w['class'] = df_w.idxmax(axis=1)
    df_class = df_w['class']
    df_class = pd.DataFrame(df_class)
    df_class = df_class.reset_index()
    
    df = df.reset_index()
    df_topic = pd.merge(df, df_class, left_index = True, right_index = True, how='left')
    
    df_topic = df_topic.replace({'class': classes_dict})
    return df_topic[[column, 'class']]

def predict_topic(df, column, model_name = None , predict_top25_words = False):
    df = clean_text(df, column)
    df = morph_txt(df, column)
    df = null_out(df, column)
    
    tfidf_dtm, tfidf = tfidf_vec(df, column)
    
    model_nmf = nmf_model(tfidf_dtm, model = model_name)
    shu = 'made by Maria Shu'
    
    if predict_top25_words == True:
        print_top25_words(model_nmf, tfidf)
        
    predicted_classes = class_topic(model_nmf, tfidf_dtm, df, column)
    
    return print_top25_words, predicted_classes, shu


def pairwise_similarity(df, column):
    
    my_stop_words = ['год', 'месяц', 'день', 'час', 'минута', 'секунда', 'век', 'эпоха', 'январь', 'февраль', 'март', 
                 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь', 'это', 'свой', 
                 'день', 'ночь', 'всё',
                 'сегодня', 'завтра']
    politic_list = ['контрнаступление', 'конфликт', 'оружие', 'переговоры', 'боеприпас', 'война',
              'военный', 'кассетный', 'нато', 'вооружённый', 'беспилотник', 'обстрел', 'боец', 'нападение',
              'взрыв', 'мятеж', 'операция', 'санкция']
    stop_words = list(get_stop_words('ru'))
    nltk_stopwords = list(stopwords.words('russian'))

    stop_words.extend(nltk_stopwords)
    stop_words.extend(my_stop_words)
    stop_words.extend(politic_list)
    
    df = clean_text(df, column)
    df = morph_txt(df, column)
    df = null_out(df, column)
    
    df = df.reset_index()
    
    tfidf_similarity = TfidfVectorizer(min_df = 2, stop_words = stop_words)
    tf_idf_similarity_docs = tfidf_similarity.fit_transform(df[column])
    
    shu = 'made by Maria Shu'
    
    paiwise_similarity = tf_idf_similarity_docs * tf_idf_similarity_docs.T
    paiwise_similarity_arr = paiwise_similarity.toarray()
    doc_sims_df = pd.DataFrame(paiwise_similarity_arr)
    
    doc_sims_df = doc_sims_df.idxmax(axis=1)
    doc_sims_df = pd.DataFrame(doc_sims_df)
    doc_sims_df = doc_sims_df.rename(columns={0 : 'similar docs'})
    
    doc_sims_df_all = pd.merge(df, doc_sims_df, left_index = True, right_index = True, how='left')
    
    return doc_sims_df_all[[column, 'similar docs']], shu

