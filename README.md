# Senticle Project

**SENTICLE**은 **SEN**timental Analysis와 ar**TICLE**을 합친 단어로써 뉴스 기사 감성분석을 위한 프로젝트입니다. 

~~포항공대 인공지능연구원(구 정보통신연구소) "**AI BigData 인재양성 심화과정 4기**" 교육 과정 수료 중 진행했던 프로젝트입니다.~~ (업데이트 완료로 인해 기존 코드를 삭제하였습니다.) 기업별 뉴스 기사를 통한 주가 상/하락의 예측을 위해 진행하였습니다. 

**Updates**

- **2021.11.09**
	- 기존 코드(2018.09~2018.10에 작성한) 전체 삭제
		- tensorflow v1 으로 작성된 코드 유지 및 가용성이 없다고 판단하여 삭제
	- Bert 계열 pretrained 모델을 사용하기 위한 Baseline으로 재구현
		- [박장원](https://github.com/monologg)님이 공개한 [KoBigBird](https://github.com/monologg/KoBigBird) pretrained 모델을 사용하였습니다.



## Project description

1. Crawler를 통한 뉴스 크롤링 진행
	- `finane_crawler.py` or `query_crawler.py`
2. `senticle-BigBird/preprocessing.py`를 통한 labeling 진행
3. `senticle-BigBird/main.py` 를 통한 학습 시작.



**자세한 내용은 [naver_crawler](https://github.com/ydy8989/senticle-proj/tree/master/naver_crawler/README.md)와 [senticle-proj](https://github.com/ydy8989/senticle-proj/tree/master/senticle-BigBird/README.md)에서 확인할 수 있습니다.**

