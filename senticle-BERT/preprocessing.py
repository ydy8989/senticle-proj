import datetime
import warnings
from glob import glob
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings(action='ignore')


def del_same(df, str_threshold=50):
    '''
    In text sorted using datetime, if the Nth text and
    the first "str_threshold" string of the (N+1)th text are exactly the same, delete one of them.
    '''
    count = 0
    leng_df = len(df)
    error_ind_lst = []
    #
    while True:
        count += 1

        try:
            print(f"Loop {count}-th preprocessing....")
            for i in range(0, len(df) - 1):
                if df.iloc[i].text[:str_threshold] == df.iloc[i + 1].text[:str_threshold]:
                    df = df.drop(df.iloc[i + 1].name, axis=0)
        except TypeError as TE:
            print(TE, f'// TE in {i} column')
            error_ind_lst.append(i)
            df = df.drop(error_ind_lst)
        except IndexError as IE:
            print(IE, f'// IE in {i} column')
            error_ind_lst.append(i)
            df = df.drop(error_ind_lst)
            pass
        else:
            print(f'-----DELETE {leng_df - len(df)}/{leng_df} SAME TEXT SUCCESS!!\n')
            return df  # break
#         print(len(df),'------------------------')
def del_similar(df, tfidf_threshold):

    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df.text)
    doc_similarities = (tfidf_matrix * tfidf_matrix.T)
    tf_idf = doc_similarities.toarray()
    del_index_list = []
    for i in range(0, len(tf_idf)):
        for j in range(i + 1, len(tf_idf)):
            if tf_idf[i][j] > tfidf_threshold:
                del_index_list.append(i)
                del_index_list.append(j)
    new_ind_lst = list(set(range(0, len(tf_idf))) - set(del_index_list))
    print(f'-----DELETE {len(set(del_index_list))}/{len(df)} SIMILAR TEXT SUCCESS!! Left : {len(new_ind_lst)}\n')
    df = df.iloc[new_ind_lst]
    df = df.reset_index(drop=True)

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


def stock_to_Label(x):
    news_date = datetime.datetime(x.year, x.month, x.day)
    while news_date not in stock.index:  # 다음날 뉴스 매칭일이 주가 정보날짜에 없으면?
        news_date = news_date + datetime.timedelta(1)
    stock_change = stock.loc[news_date].Change
    if stock_change >= 0:
        return 1
    else:
        return 0


def labeling(df):
    df['market_time'] = df.time.apply(lambda x: datetime.datetime(x.year, x.month, x.day) + datetime.timedelta(1)
    if datetime.time(x.month, x.day) >= datetime.time(hour=15, minute=30)
    else datetime.datetime(x.year, x.month, x.day))
    df['label'] = df['market_time'].apply(stock_to_Label)
    return df


if __name__ == '__main__':
    datasets = '../data/*.csv'
    dirs = Path(datasets)
    filename = glob(datasets)[-1]
    print(filename)
    df = pd.read_csv(filename, names=['time', 'text'], encoding='cp949', header=None, error_bad_lines=False,
                     index_col='time')

    drop_lst = []
    for i in range(len(df)):
        if len(df.index[i]) != 17 and len(df.index[i].split('.')[0]) != 4:
            drop_lst.append(df.index[i])

    df = df.drop(drop_lst)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    df = df.sort_index()
    df = df.reset_index()
    df = del_same(df)
    df = del_similar(df, tfidf_threshold=0.7)
    df = srt_end_cutting(df)
    # print(len(df))

    stock = fdr.DataReader(filename[-10:-4],
                           f'{df.time[0].year}-{df.time[0].month}-{df.time[0].day}',
                           f'{df.time[len(df) - 1].year}-{df.time[len(df) - 1].month}-{df.time[len(df) - 1].day}')

    # #Change가 0인 행 모두 삭제 (나중에 병합시 nan값으로 만들고, 채워주기 위함)
    stock = stock.drop(stock[stock.Change == 0].index, axis=0)
    df = labeling(df)

