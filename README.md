# 💬 Senticle(SENimental analysis + arTICLE) Project

## 📚Project Description

기준 시점 뉴스 데이터를 통해 다음 날 주가 상/하락을 예측하는 프로젝트입니다.

- ***기간*** : 2018.09.03 ~ 2018.10.16(약 6주)
- ***task description*** :
	- `Input` : 특정 기업 뉴스데이터 약 4년치
	- `Output` : 주가 상/하락에 대한 binary classification 정보 
- ***data overview*** :
	- 

## 1. crawler

#### bigkinds_crawler.py

- ~~뉴스 데이터 제공 사이트 [Bigkinds](https://www.kinds.or.kr)로부터 keyword를 포함/배제한 기사 수집 크롤러~~
- 현재 수정 요망. 사이트 개편으로 인한 작동 중지
- Selenuim 혹은 타 사이트에서의 크롤링 방식을 알아보는 중
    - Bigkinds의 경우 api는 협약을 맺은 기관에만 제공하는 방향으로 개편됨
    - [링크](https://www.kinds.or.kr/news/qnaView.do)
#### naver_crawler.py
- 네이버 증권 뉴스로부터 크롤링

# 2. senticle-BERT
- **구현중...**
- Pipeline
    - `crawler/naver_crawler.py`로부터 크롤링 
        - 크롤링된 `csv`파일은 `senticle-proj/data/`에 저장
    - `preprocessing.py`를 통해 `.csv` 파일 전처리
    - `basic_kobert.py`를 통해 학습 

## ~~3. Senticle-CNN~~
#### contents
1. [cnn_tool.py](https://github.com/ydy8989/senticle/blob/master/Senticle-CNN/cnn_tool.py)
    - main.py에서 사용하는 자연어 전처리 관련 함수들 
2. [crawler.py](https://github.com/ydy8989/senticle/blob/master/Senticle-CNN/crawler.py)
    - 실시간으로 뉴스기사를 크롤링해 서버에 저장
3. [final_preprecess.py](https://github.com/ydy8989/senticle/blob/master/Senticle-CNN/final_preprecess.py)
    - XXX_crawler.py를 통해 얻은 raw_data를 정제 
4. [main.py](https://github.com/ydy8989/senticle/blob/master/Senticle-CNN/main.py)
    - TextCNN 모델

5. [make_preprocess.py](https://github.com/ydy8989/senticle/blob/master/Senticle-CNN/make_preprocess.py)
    - soynlp를 활용한 형태소 분석 및 불용어 처리
    
6. [train.py](https://github.com/ydy8989/senticle/blob/master/Senticle-CNN/train.py)
    - 트레이닝
    - Flag를 이용해 파라미터 지정 
7. [Senticle-LSTM.py](https://github.com/ydy8989/senticle-proj/blob/master/senticle-CNN/Senticle-LSTM.py)
    - LSTM, GRU, BasicRNN 셀을 이용한 결과에 신빙성 부여


### 안드로이드 소스코드(java)
https://github.com/GeonKim/android_pospirl

