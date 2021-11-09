## Senticle-BigBird

### 디렉토리 구조

```
Senticle-BigBird>
├──config.yml
├──kobert_tokenization.py
├──load_data.py
├──loss.py
├──main.py
├──preprocessing.py
└──README.md

```



#### File overview

- `config.yml` : parameter 변경을 위한 config 파일
- `kobert_tokenization.py`  : Kobert 사용을 위한 토크나이저 파일
- `load_data.py` : 데이터로더를 위한 토크나이저 함수 + 데이터셋 클래스 
- `loss.py` : loss 함수
- `main.py` : 메인 학습 파일
- `preprocessing.py` : 레이블링
	- Finance-DataReader 라이브러리를 통해 익일 주가 상/하락 정보 label로 mapping
	- summarize 옵션 추가 - pororo
	- `input` : 뉴스 크롤링 csv파일 원본
	- `output` : 레이블링 완료(option : summarize)된 csv파일



### How to use

```sh
$ python main.py --config_file_path='./config.yml' --config='bigbird' --data_path='../data/pre_005930.csv'
```

- optimizer, shceduler, learing rate 등을 추가하고 싶으면 `config.yml` 수정 후 `main.py`에 추가하면 됩니다. 

