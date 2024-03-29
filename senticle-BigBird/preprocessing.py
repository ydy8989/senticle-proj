import datetime
import warnings
from glob import glob
from pororo import Pororo
import FinanceDataReader as fdr
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from naver_crawler.exceptions import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
# warnings.filterwarnings(action='ignore')


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
        if count % 10 == 0:
            print(count)
            try:
                print(f"Loop {count}-th preprocessing....")
                for i in range(0, len(df) - 1):
                    print(i)
                    if df.iloc[i].text[:str_threshold] == df.iloc[i + 1].text[:str_threshold]:
                        df = df.drop(df.iloc[i + 1].name, axis=0)
            except TypeError as TE:
                print(TE, f'// TypeError in [{i}] row')
                error_ind_lst.append(i)
                df = df.drop(error_ind_lst)
                error_ind_lst = []
            except IndexError as IE:
                print(IE, f'// IndexError in [{i}] row')
                error_ind_lst.append(i)
                df = df.drop(error_ind_lst)
                error_ind_lst = []
            #             pass
        else:
            print(f'-----DELETE {leng_df - len(df)}/{leng_df} SAME TEXT SUCCESS!!-----')
            return df  # break


def del_similar(df, tfidf_threshold):
    #
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
    print(f'-----DELETE {len(set(del_index_list))}/{len(df)} SIMILAR TEXT SUCCESS!!-----')
    print(f'Left : {len(new_ind_lst)} lines')
    df = df.iloc[new_ind_lst]
    df = df.reset_index(drop=True)
    return df


def srt_end_cutting(df):
    # 시작일 기준 3시 30분 전 데이터 삭제
    try:
        df = df.drop(df[df.time <= datetime.datetime(year=df.time[0].year,
                                                     month=df.time[0].month,
                                                     day=df.time[0].day,
                                                     hour=15,
                                                     minute=30)].index, axis=0)
    except:
        pass

    # 종료일 기준 3시 30분 후 데이터 삭제
    try:
        df = df.drop(df[df.time >= datetime.datetime(year=df.time[len(df) - 1].year,
                                                     month=df.time[len(df) - 1].month,
                                                     day=df.time[len(df) - 1].day,
                                                     hour=15, minute=30)].index, axis=0)
    except:
        pass
    df = df.reset_index(drop=True)
    return df


def stock_to_Label(x):
    news_date = datetime.datetime(x.year, x.month, x.day)
    # print(news_date, stock.index)
    if news_date <= stock.index[-1]:
        while news_date not in stock.index:  # 다음날 뉴스 매칭일이 주가 정보날짜에 없으면?
            news_date = news_date + datetime.timedelta(1)
        stock_change = stock.loc[news_date].Change
        if stock_change >= 0:
            return 1
        elif stock_change < 0:
            return 0
    else:
        return 2


def labeling(df):
    df['market_time'] = df.time.apply(
        lambda x: datetime.datetime(x.year, x.month, x.day) + datetime.timedelta(1)
        if datetime.time(x.month, x.day) >= datetime.time(hour=15, minute=30)
        else datetime.datetime(x.year, x.month, x.day))
    df['label'] = df['market_time'].apply(stock_to_Label)
    no_label_ind = df[df['label'] == 2].index
    del_df = df.drop(no_label_ind)
    print(del_df)
    return del_df


def time_transform(x):
    if '오후' in x:
        if int(x.split(' ')[-1].split(':')[0]) == 12:
            return ' '.join(x.split('. 오후 '))
        else:
            newtime = int(x.split(' ')[-1].split(':')[0]) + 12
            return x.split('. 오후 ')[0] + ' ' + str(newtime) + x.split('. 오후 ')[1][-3:]
    else:
        return ' '.join(x.split('. 오전 '))

def total_preprocess(filename, colnames, tfidf_threshold, query_crawler=True):
    drop_lst = []
    df = pd.read_csv(filename,
                     names=colnames,
                     encoding='utf8',
                     header=None,
                     # error_bad_lines=False,
                     on_bad_lines='skip',
                     index_col='time')
    if query_crawler:
        try:
            stocklist = fdr.StockListing('KRX')
            # print(df.stock)
            StockCode = stocklist[stocklist['Name'] == df.stock[0]].Symbol.values.tolist()[0]
            print('StockCode : ', StockCode)
        # break
        except:
            raise NoStockSymbol(df.stock[0])
        df.index = df.index.map(lambda x: time_transform(x))

    for i in range(len(df)):
        # 날짜 파싱 시 finance 크롤러는 앞에 한 칸이 들어감 => 17
        # 일반 query 크롤러는 한 칸이 미포함 => 16
        # and 뒷 부분 : 가끔 년도가 4자리 이외의 숫자로 크롤링될 때가 있음.
        if ((len(df.index[i]) != 16) or (len(df.index[i]) != 17)) and len(df.index[i].split('.')[0]) != 4:
            drop_lst.append(df.index[i])
    df = df.drop(drop_lst)
    df.index = pd.to_datetime(df.index,
                              format='%Y-%m-%d %H:%M:%S',
                              errors='coerce')
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    df = df.sort_index()
    df = df.reset_index()
    df = del_same(df)
    ##########################################################
    # tfidf 처리 과정에서 메모리가 부족할 수 있습니다. (del_similar)
    ##########################################################
    # df = del_similar(df, tfidf_threshold)
    df = srt_end_cutting(df)

    if query_crawler:
        return df, StockCode
    else:
        return df
def summarize_text(df, summarize_model):
    lst=[]
    summary = Pororo(task="summary", model=summarize_model, lang="kr")
    for i in tqdm(range(len(df))):
        lst.append(summary(df.text[i]))
    df['summarize'] = lst
    return df

if __name__ == '__main__':
    tfidf_threshold = 0.75 # 연속하는 두 문서의 유사도가 tfidf_threshold 이상이면 삭제
    query_crawler = True # False means that crawl data is output of finance_crawler.py
    datasets = '../data/삼성전자_*.csv' # ./data/query_start_end.csv 파일
    ###########################
    #        summarize
    ###########################
    do_summarize = False
    summarize_model = 'abstractive' #'abstractive'


    for filename in glob(datasets):
        pathname = filename.split('/')[-1].split('.')[0]
        # break
        # naver-Finane crawl
        if not query_crawler:
            StockCode = filename.split('.csv')[0].split('_')[-1]
            print('-' * 100)
            print('-' * 100)
            print(f'FILENAME : {filename}')
            print(f'StockCode : {StockCode}')
            print(f'tfidf_threshold : {tfidf_threshold}')
            print('-' * 50)
            colnames = ['time','text']
            df = total_preprocess(filename, colnames, tfidf_threshold)
            stock = fdr.DataReader(StockCode,
                                   f'{df.time[0].year}-{df.time[0].month}-{df.time[0].day}',
                                   f'{df.time[len(df) - 1].year}-{df.time[len(df) - 1].month}-{df.time[len(df) - 1].day}')

            # #Change가 0인 행 모두 삭제 (나중에 병합시 nan값으로 만들고, 채워주기 위함)
            stock = stock.drop(stock[stock.Change == 0].index, axis=0)
            df = labeling(df)
            if do_summarize:
                print('-'*100)
                print('Start Summarization')
                print('-' * 100)
                df = summarize_text(df,summarize_model)
            df.to_csv(f'../data/pre_{pathname}.csv')
        else:
            # naver-query based crawl
            print('-' * 100)
            print('-' * 100)
            print(f'FILENAME : {filename}')
            print(f'tfidf_threshold : {tfidf_threshold}')
            print('-' * 50)
            colnames = ['time', 'stock', 'newspaper', 'header', 'text', 'url']
            df, StockCode = total_preprocess(filename, colnames, tfidf_threshold)
            stock = fdr.DataReader(StockCode,
                                   f'{df.time[0].year}-{df.time[0].month}-{df.time[0].day}',
                                   f'{df.time[len(df) - 1].year}-{df.time[len(df) - 1].month}-{df.time[len(df) - 1].day}')
            # #Change가 0인 행 모두 삭제 (나중에 병합시 nan값으로 만들고, 채워주기 위함)
            stock = stock.drop(stock[stock.Change == 0].index, axis=0)
            df = labeling(df)
            if do_summarize:
                print('-'*100)
                print('Start Summarization')
                print('-' * 100)
                df = summarize_text(df,summarize_model)
                print('Summarization END')
            df.to_csv(f'../data/pre_{pathname}.csv')