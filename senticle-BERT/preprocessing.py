from glob import glob
from pathlib import Path
import os

datasets = 'new_senticle/NewsData/*.csv'
# pathlib

dirs = Path(datasets)
path = dirs / '*.csv'
# print(datasets)
filename = glob(datasets)[0]
print(filename)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime

df = pd.read_csv(filename,  names = ['time','text'], header=None, index_col='time')

# str to datetime
df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='raise')
df = df.sort_index()
df = df.reset_index()
df


def del_same(df, str_threshold=50):
    '''
    In text sorted using datetime, if the Nth text and the first "str_threshold" string of the (N+1)th text are exactly the same, delete one of them.
    '''
    count = 0
    while True:
        count += 1
        try:
            print(f"Loop {count}-th preprocessing....")
            for i in range(0, len(df) - 1):
                if df.iloc[i].text[:str_threshold] == df.iloc[i + 1].text[:str_threshold]:
                    df = df.drop(df.iloc[i + 1].name, axis=0)
        except IndexError:
            pass
        else:
            print('-----DELETE SAME TEXT SUCCESS!!\n')
            return df  # break


def del_similar(df):
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df.text)
    doc_similarities = (tfidf_matrix * tfidf_matrix.T)
    tf_idf = doc_similarities.toarray()
    del_index_list = []
    for i in range(0, len(tf_idf)):
        for j in range(i + 1, len(tf_idf)):
            if tf_idf[i][j] > 0.5:
                print(f'{i}-th and {j}-th text are similar! TF-IDF score is {round(tf_idf[i][j], 3)}')
                del_index_list.append(i)
                del_index_list.append(j)
    new_ind_lst = list(set(range(0, len(tf_idf))) - set(del_index_list))
    df = df.iloc[new_ind_lst]
    df = df.reset_index(drop=True)
    print('-----DELETE SIMILAR TEXT SUCCESS!!\n')
    return df


def srt_end_cutting(df):
    # 시작일 기준 3시 30분 전 데이터 삭제
    try:
        df = df.drop(df[df.time <= datetime.datetime(year=df.time[0].year, month=df.time[0].month, day=df.time[0].day,
                                                     hour=15, minute=30)].index, axis=0)
    except:
        pass

    # 종료일 기준 3시 30분 후 데이터 삭제
    try:
        df = df.drop(df[df.time >= datetime.datetime(year=df.time[len(df) - 1].year, month=df.time[len(df) - 1].month,
                                                     day=df.time[len(df) - 1].day, hour=15, minute=30)].index, axis=0)
    except:
        pass
    df = df.reset_index(drop=True)
    return df


df = del_same(df)
df = del_similar(df)
df = srt_end_cutting(df)
df.head()