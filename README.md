# Senticle Project


## 1. install requirements
```pip install -r requirements.txt```

## 2. crawler
#### bigkinds_crawler.py
- 뉴스 데이터 제공 사이트 [Bigkinds](https://www.kinds.or.kr)로부터 keyword를 포함/배제한 기사 수집 크롤러
#### naver_crawler.py
- 네이버 증권 뉴스로부터 크롤링


## 3. Senticle-CNN
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
7. [Senticle-LSTM.py]()
    - LSTM, GRU, BasicRNN 셀을 이용한 결과에 신빙성 부여


### 안드로이드 소스코드(java)
https://github.com/GeonKim/android_pospirl


