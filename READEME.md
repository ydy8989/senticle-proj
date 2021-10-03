# Senticle Project

**SENTICLE**은 **SEN**timental Analysis와 **ARTI**cle을 합친 단어로써 뉴스 기사 감성분석을 위한 프로젝트입니다. 

포항공대 인공지능연구원(구 정보통신연구소) "AI BigData 인재양성 심화과정 4기" 교육 과정 수료 중 진행했던 프로젝트입니다. 기업별 뉴스 기사를 통한 주가 상/하락의 예측을 위해 진행하였습니다. 



## 0. Updates

**2021.09**

- 크롤링 파일 재구현

  - 기존 Bigkinds가 더이상 크롤링이 불가능하게 됨 > 동적 
  - selenium으로 하기에는 너무 양이 많고 오래걸림. 
  - `naver_crawler.py`는 네이버 증권에서 특정 기업의 뉴스공시 탭 기사들만 크롤링
    - 양이 너무 적음
  - 새로 만든 `naver_crawler_query.py`는 검색어 기반 + 네이버 뉴스로 다시 파싱한 기사들만 크롤링 

  

- Transformer 계열로 업데이트 진행중..

**2018.09**

- 1DCNN을 활용한 text classification 모델 구현
- 토크나이저 : soynlp
- 임베딩 : randomVectorization
  - 문장 분류와 달리 document 분류였기에 Fasttext 혹은 word2vec의 성능이 더 안좋았음.



## 1. crawler

~~`bigkinds_crawler.py`~~

- ~~뉴스 데이터 제공 사이트 [Bigkinds](https://www.kinds.or.kr)로부터 keyword를 포함/배제한 기사 수집 크롤러~~
- bigkinds는 더 이상 크롤링이 되지 않고 api를 사용하게끔 바뀌었으므로 파일을 삭제하였습니다. 

`naver_crawler.py`

- 네이버 금융 `뉴스·공시` 탭으로부터 크롤링
- ![image](https://user-images.githubusercontent.com/38639633/134671102-9f6d0b7c-b027-462c-a0d1-c20c8e3f5b95.png)

`query_crawler.py`

- 검색 기간과 검색어를 기반으로 뉴스 크롤링
- 네이버는 검색기간을 아무리 길게 설정해도 최대 400페이지만 검색 가능
- 최신순으로 검색되는 뉴스 페이지가 400페이지 도달시 마지막 기사 날짜를 파싱 후 해당 날짜로부터 다시 최신순으로 검색
- 뉴스 기사는 언론사별 html 구조가 상이하므로 `네이버뉴스`에서 재지원하는 기사만 크롤링





## 2. Senticle-BERT

`preprocessing.py`

- Finance-datareader 라이브러리로부터 크롤링된 뉴스 기사의 다음날 주가 상하락 정보 레이블링

## 3. Senticle-CNN

1. `cnn_tool.py`
    - main.py에서 사용하는 자연어 전처리 관련 함수들 

