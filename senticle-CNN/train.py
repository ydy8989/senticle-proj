from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import csv
import hashlib
import threading
import time
from multiprocessing import Process


# print('get_last_page 시작')
def get_last_page(page=1):
    global code
    if title_entity == True:
        # 제목 기준 crawler
        main_url = 'https://finance.naver.com/item/news_news.nhn?code='+code+'&page=' + str(
            page) + '&sm=title_entity_id.basic&clusterId='
    else:
        # 내용 기준 crawler
        main_url = 'https://finance.naver.com/item/news_news.nhn?code='+code+'&page=' + str(
            page) + '&sm=entity_id.basic&clusterId='
    soup = get_html(main_url)
    last_btn = soup.find('td', class_='pgRR')
    if last_btn is None:
        return page
    else:
        last_link = last_btn.a['href']
        split1 = last_link.split('=')
        split2 = split1[2].split('&')
        last_page = split2[0]
        return get_last_page(last_page)

def get_html(url):
    html = urlopen(url)
    # print(html)
    soup = BeautifulSoup(html, 'html.parser', from_encoding='cp949')
    # print(soup)
    return soup

def get_url(page):
    global code
    global title_entity

    main_url = 'https://finance.naver.com/item/news_news.nhn?code=' + code + '&page=' + str(page) + '&sm=title_entity_id.basic&clusterId='
    print(main_url)
    soup = get_html(main_url)

    table = soup.find('table', attrs={'class': 'type5'})

    test_list = []
    #리눅스 버전
    # class=relation_lst로 시작하는 행을 삭제
    for tag in table.select('tr[class^="relation_lst"]'):
        tag.decompose()

    # for문을 돌면서 테이블에서 값을 가져옴
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) == 3:
            # href = 뉴스 기사 주소
            # cells[1] = 신문사 이름
            # cells[2] = 뉴스 입력 시간
            # href_list.append(cells[0].a['href'])
            # date_list.append(cells[2].text)
            test_list.append([cells[2].text, cells[0].a['href']])

    return test_list


# def realtime_crawler():
#     code = '005490'
#     main_url = 'https://finance.naver.com/item/news_news.nhn?code=' + code + '&page=1&sm=title_entity_id.basic&clusterId='
#     soup = get_html(main_url)
#     table = soup.find('table', attrs={'class': 'type5'})
#     test_list = []
#     # 리눅스 버전
#     # class=relation_lst로 시작하는 행을 삭제
#     for tag in table.select('tr[class^="relation_lst"]'):
#         tag.decompose()
#     # for문을 돌면서 테이블에서 값을 가져옴
#     for row in table.find_all('tr'):
#         cells = row.find_all('td')
#         title = row.find_all('td',{'class':'title'})
#         if len(cells) == 3:
#             # href = 뉴스 기사 주소
#             # cells[1] = 신문사 이름
#             # cells[2] = 뉴스 입력 시간
#             # href_list.append(cells[0].a['href'])
#             # date_list.append(cells[2].text)
#             test_list.append([cells[2].text, cells[0].text, cells[0].a['href']])
#
#     return test_list

def read_article_url(l):
    news_url = 'https://finance.naver.com'

    article_text = []

    if l != None:
        article_url = news_url + l

        html = get_html(article_url)

        news_read = html.find('div', class_='scr01')

        for tag in news_read.select('div[class=link_news]'):
            tag.decompose()

        for tag in news_read.select('ul'):
            tag.decompose()

        for tag in news_read.select('strong'):
            tag.decompose()

        for tag in news_read.select('table'):
            tag.decompose()

        for tag in news_read.select('a'):
            tag.decompose()

        text = news_read.text

        text2 = text.split('.')

        for s in range(len(text2)-1, 0, -1):
            if '@' in text2[s]:
                for j in range(len(text2) - 1, s - 1, -1):
                    text2.pop(j)

        for s in range(len(text2) - 1, 0, -1):
            if '한경로보뉴스' in text2[s]:
                for j in range(len(text2) - 1, s - 1, -1):
                    text2.pop(j)

        text3 = ('.').join(text2)

        text4 = re.sub('\[.+?\]', '', text3, 0).strip()

        article_text.append([l[0], text4 + "."])


        f = open('POSCO.csv', 'a', encoding='UTF-8', newline='')
        wr = csv.writer(f)
        wr.writerow(article_text)
        return text4 + "."
    else:
        return 0

def read_article_main(l,code, title_entity):
    news_url = 'https://finance.naver.com'
    article_text = []
    if l != None:
        article_url = news_url + l[1]
        html = get_html(article_url)
        news_read = html.find('div', class_='scr01')
        for tag in news_read.select('div[class=link_news]'):
            tag.decompose()
        for tag in news_read.select('ul'):
            tag.decompose()
        for tag in news_read.select('strong'):
            tag.decompose()
        for tag in news_read.select('table'):
            tag.decompose()
        for tag in news_read.select('a'):
            tag.decompose()
        text = news_read.text
        text2 = text.split('.')
        for s in range(len(text2)-1, 0, -1):
            if '@' in text2[s]:
                for j in range(len(text2) - 1, s - 1, -1):
                    text2.pop(j)
        for s in range(len(text2) - 1, 0, -1):
            if '한경로보뉴스' in text2[s]:
                for j in range(len(text2) - 1, s - 1, -1):
                    text2.pop(j)
        text3 = ('.').join(text2)
        text4 = re.sub('\[.+?\]', '', text3, 0).strip()
        article_text.append([l[0], text4 + "."])
        print(article_text)
        if title_entity:
            f = open(f'../data/title_{code}.csv', 'a', encoding='UTF-8', newline='')
            wr = csv.writer(f)
            wr.writerow(article_text)
            f.close()
        else:
            f = open(f'../data/con_{code}.csv', 'a', encoding='UTF-8', newline='')
            wr = csv.writer(f)
            wr.writerow(article_text)
            f.close()
    else:
        return 0

if __name__ == '__main__':
    code = "005930"#str(input('Input Stock Item Code :'))
    tmp = False
    # True : 제목 기준 클러스터링 False : 내용 기준 검색
    while not tmp:
        ans = 'y'  # str(input('뉴스 공시 정렬 기준 (제목(Y), 내용(N)) : ')).lower()
        if ans == 'y':
            title_entity = True
            tmp = True
        elif ans == 'n':
            title_entity = False
            tmp = True
        else:
            print('정확히 입력해주세요.')
            tmp = False

    last_page = min(1, int(get_last_page())) + 1 # page 401이상 있어도 기사가 안바뀜
    print("last page # : ",last_page)
    for i in range(1, last_page):
        test = get_url(i)
        procs = []
        for index, number in enumerate(test):
            proc = Process(target=read_article_main, args=(number, code, title_entity))
            procs.append(proc)
            proc.start()
            # print(number[0])

        for proc in procs:
            proc.join()

    h = hashlib.md5()
    h.update('/item/news_read.nhn?article_id=0004238328&amp;office_id=009&amp;code=005490&amp;page=1&amp;sm=title_entity_id.basic'.encode())
    print(h.hexdigest())
    # while True:
    #     for i in realtime_crawler():
    #         i.append(read_article_url(i[2]))
    #         print(i)
    #     break
    #     # time.sleep(5)


