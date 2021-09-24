import os
import platform
import calendar
import requests
import re
from time import sleep
from bs4 import BeautifulSoup
from multiprocessing import Process
import datetime
from crawler.exceptions import *
from crawler.articleparser import ArticleParser
from crawler.writer import Writer
from tqdm import tqdm
class ArticleCrawler(object):
    def __init__(self):
        self.selected_queries = []
        self.user_operating_system = str(platform.system())

    def set_category(self, *args):
        self.selected_queries = args

    def set_date_range(self, ds, de):#datestart, dateend
        self.ds = ds
        self.de = de
        print(self.ds, self.de)


    @staticmethod
    def get_url_data(url, max_tries=5):
        remaining_tries = int(max_tries)
        while remaining_tries > 0:
            try:
                return requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            except requests.exceptions:
                sleep(1)
            remaining_tries = remaining_tries - 1
        raise ResponseTimeout()

    def make_news_page_url_my(self, category_url, ds, de):
        made_urls = []

        '''
        1. 10000추가해서 400있는지 확인
        2. 있으면 마지막 날짜 파싱
        3. ds부터 마지막 날짜까지 픽스하고 1~400으로 링크 추가 
        4. 날짜 new de로 업데이트 
        5. 1부터 4 반복'''
        # de와 ds는 한번에 받는게 좋아보인다.
        while True:
            url = category_url + f'ds={ds}&de={de}' \
                f'&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,' \
                f'p:from{ds}to{de},a:all&start='
            totalpage = ArticleParser.find_news_totalpage(url + "10000")  # 어떨땐 400, 아닐땐 1~399중 하나
            request = self.get_url_data(url + f"{totalpage - 1}1")  # 마지막 페이지
            document = BeautifulSoup(request.content, 'html.parser')
            last_time_post = document.select('.api_subject_bx .group_news li .news_area .info_group span.info')
            if totalpage == 400 and len(last_time_post) >= 1:
                last_time = datetime.datetime.strptime(last_time_post[-1].text, '%Y.%m.%d.')
                for page in range(1, totalpage+1):
                    made_urls.append(url + f'{page-1}1')
                # 다했으면 de 날짜 바꾸기
                if last_time > datetime.datetime.strptime(ds, '%Y%m%d'):
                    de = last_time - datetime.timedelta(1)
                    de = de.strftime('%Y%m%d')
                    print(f'바뀔 시작 : {ds}, 종료 : {de}')
                else:
                    break

            elif totalpage != 400:
                print(totalpage - 1, '막지막 여기페이지는 여기올')  # totalpage가 400이 아니라 그 미만일 때
                for page in range(1, totalpage):
                    made_urls.append(url + f'{page}1')

                print('-' * 100)
                print('최종 크롤링 페이지 수 :', len(made_urls))
                print('-' * 100)
                break
            print('누적 페이지 수:', len(made_urls))
        return made_urls

    def crawling(self, query):
        # Multi Process PID
        print(query + " PID: " + str(os.getpid()))

        writer = Writer(query, ds=self.ds, de=self.de)
        url_format = f'https://search.naver.com/search.naver?where=news&query=' \
                     f'{query}&sm=tab_opt&sort=1&photo=0&field=0&pd=3&'
        target_urls = self.make_news_page_url_my(url_format, self.ds, self.de)
        print(query + " Urls are generated")
        print("The crawler starts")
        print(len(target_urls))
        for url in tqdm(target_urls):
            request = self.get_url_data(url)
            document = BeautifulSoup(request.content, 'html.parser')
            temp_post = document.select("a[href^='https://news.naver.com/main/read']")
            # 각 페이지에 있는 기사들의 url 저장
            post_urls = []
            for line in temp_post:
                # 해당되는 page에서 모든 기사들의 URL을 post_urls 리스트에 넣음
                post_urls.append(line.attrs['href'])
            # print(post_urls)
            del temp_post
            for content_url in post_urls:  # 기사 url
                # 크롤링 대기 시간
                sleep(0.01)
                # print(content_url)
                # 기사 HTML 가져옴
                request_content = self.get_url_data(content_url)

                try:
                    document_content = BeautifulSoup(request_content.content, 'html.parser')
                except:
                    continue

                try:
                    # 기사 제목 가져옴
                    tag_headline = document_content.find_all('h3', {'id': 'articleTitle'}, {'class': 'tts_head'})
                    # 뉴스 기사 제목 초기화
                    text_headline = ''
                    text_headline = text_headline + ArticleParser.clear_headline(
                        str(tag_headline[0].find_all(text=True)))
                    # 공백일 경우 기사 제외 처리
                    if not text_headline:
                        continue
                    # 기사 본문 가져옴
                    tag_content = document_content.find_all('div', {'id': 'articleBodyContents'})
                    # 뉴스 기사 본문 초기화
                    text_sentence = ''
                    text_sentence = text_sentence + ArticleParser.clear_content(str(tag_content[0].find_all(text=True)))
                    # 공백일 경우 기사 제외 처리
                    if not text_sentence:
                        continue
                    # print(text_sentence)
                    # 기사 언론사 가져옴
                    tag_company = document_content.find_all('meta', {'property': 'me2:category1'})

                    # 언론사 초기화
                    text_company = ''
                    text_company = text_company + str(tag_company[0].get('content'))

                    # 공백일 경우 기사 제외 처리
                    if not text_company:
                        continue
                    # print(text_company)
                    # 기사 시간대 가져옴
                    time = re.findall('<span class="t11">(.*)</span>', request_content.text)[0]

                    # CSV 작성
                    writer.write_row([time, query, text_company, text_headline, text_sentence, content_url])

                    # del time
                    del text_company, text_sentence, text_headline
                    del tag_company
                    del tag_content, tag_headline
                    del request_content, document_content

                # UnicodeEncodeError
                except Exception as ex:
                    del request_content, document_content
                    pass
        writer.close()
    def start(self):
        # MultiProcess 크롤링 시작
        for category_name in self.selected_queries: #selected_cate : 리스트임
            proc = Process(target=self.crawling, args=(category_name,))
            proc.start()


if __name__ == "__main__":
    Crawler = ArticleCrawler()
    Crawler.set_category('삼성전자')#, '포스코','KT','검색어',...
    Crawler.set_date_range('20200101', '20200401')# 'YYYYMMDD'
    Crawler.start()