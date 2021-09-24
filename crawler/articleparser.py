import requests
import re
from bs4 import BeautifulSoup


class ArticleParser(object):
    special_symbol = re.compile('[\{\}\[\]\/?,;:|\)*~`!^\-_+<>@\#$&▲▶◆◀■【】\\\=\(\'\"]')
    content_pattern = re.compile('본문 내용|TV플레이어| 동영상 뉴스|flash 오류를 우회하기 위한 함수 추가function  flash removeCallback|tt|앵커 멘트|xa0')

    @classmethod
    def clear_content(cls, text):
        # 기사 본문에서 필요없는 특수문자 및 본문 양식 등을 다 지움
        newline_symbol_removed_text = text.replace('\\n', '').replace('\\t', '').replace('\\r', '')
        special_symbol_removed_content = re.sub(cls.special_symbol, ' ', newline_symbol_removed_text)
        end_phrase_removed_content = re.sub(cls.content_pattern, '', special_symbol_removed_content)
        blank_removed_content = re.sub(' +', ' ', end_phrase_removed_content).lstrip()  # 공백 에러 삭제
        reversed_content = ''.join(reversed(blank_removed_content))  # 기사 내용을 reverse 한다.
        content = ''
        for i in range(0, len(blank_removed_content)):
            # reverse 된 기사 내용중, ".다"로 끝나는 경우 기사 내용이 끝난 것이기 때문에 기사 내용이 끝난 후의 광고, 기자 등의 정보는 다 지움
            if reversed_content[i:i + 2] == '.다':
                content = ''.join(reversed(reversed_content[i:]))
                break
        return content

    @classmethod
    def clear_headline(cls, text):
        # 기사 제목에서 필요없는 특수문자들을 지움
        newline_symbol_removed_text = text.replace('\\n', '').replace('\\t', '').replace('\\r', '')
        special_symbol_removed_headline = re.sub(cls.special_symbol, '', newline_symbol_removed_text)
        return special_symbol_removed_headline

    @classmethod
    def find_news_totalpage(cls, url): # 10000추가되어서 들어옴
        def is_target(url, middle):  # 정답 243 o 244 x
            new_url = url[:-5] + str(middle) + '1'
            request_content = requests.get(new_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            document_content = BeautifulSoup(request_content.content, 'html.parser')
            po = document_content.select('.sc_page_inner .btn')
            return po
        # 당일 기사 목록 전체를 알아냄
        try:
            request_content = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            document_content = BeautifulSoup(request_content.content, 'html.parser')
            po = document_content.select('.sc_page_inner .btn')
            return int(po[-1].text)
        except Exception:
            print('마지막 페이지 위한 이진탐색중...')
            search_page = [i for i in range(0, 400)]
            left = 0
            right = len(search_page) - 1
            check_list = []
            while left <= right:
                mid = (left + right) // 2
                preset = (left, mid, right)
                check_list.append(preset)
                if len(is_target(url, mid)) != 0:
                    left = mid + 1
                elif len(is_target(url, mid)) == 0:
                    right = mid + 1
                nowset = (left, mid, right)
                check_list.append(nowset)
                if check_list[0]==check_list[1]:
                    break
                else:
                    check_list.pop(0)
            print('마지막 페이지 :',left-1)
            return int(left-1)