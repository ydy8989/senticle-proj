# naver_crawler

사용할 수 있는 크롤러는 두 가지 입니다. 

1. finance_crawler.py
2. query_crawler.py



### How to use

**`finance_crawler.py`**

```sh
$ python finance_crawler.py
```



- [네이버 금융](https://finance.naver.com/)의 특정 기업 `뉴스·공시` 탭으로부터 크롤링을 진행합니다. ![image](https://user-images.githubusercontent.com/38639633/140779753-a08edee8-8e52-4b1a-a14e-73469550d417.png)

	> 단, 네이버 뉴스·공시 탭의 경우, 400페이지 이상에서는 더 이상 과거 뉴스를 지원하지 않습니다.  
	> 따라서 1에서 400페이지까지만 크롤링을 진행합니다. 


**`query_crawler.py`**

```sh
$ python query_crawler.py --query='삼성전자' --start_date='20200101' --end_date='20200201'
```

- **Ubuntu에서 작성하였습니다. 윈도우 환경에서는 인코딩 문제가 발생할 수 있습니다.**

- 네이버 메인 검색창에 검색어를 기반으로 [네이버 뉴스](https://news.naver.com/) 크롤링을 진행합니다. 

  ![image](https://user-images.githubusercontent.com/38639633/140784382-1a751058-5ade-40dc-aa98-d4d36f1c2cbe.png)

  > 기사 중 위와 같은 "**네이버뉴스**" 표시가 있는 기사만 크롤링 진행합니다.  

  

  ![image](https://user-images.githubusercontent.com/38639633/140987521-22b4d9dc-3972-4dcd-abef-2873a9ddcbf5.png)

  > - '네이버뉴스' 표시가 있는 기사는 위의 "Naver 뉴스" 페이지에 올라가며, 해당 기사의 html 구조들은 모두 동일합니다. 
  > - 우측 상단 뉴스 검색에서의 검색은 네이버 메인에서의 검색과 동일합니다. 

  

- 검색 기간 무제한으로 설정 가능합니다.

	400페이지 제한이 걸려있기 때문에 400페이지씩 전체 검색 기간을 잘라가며 크롤링 합니다.([해당 부분 코드](https://github.com/ydy8989/senticle-proj/blob/9c6545fa9a36ae25d2d87acb7d6e7cc250ac1151/naver_crawler/query_crawler.py#L40-L72) 참고)

- 뉴스 기사는 언론사별 html 구조가 상이하므로 `네이버뉴스`에서 재지원하는 기사만 크롤링

	> **Q1) 왜 "네이버뉴스" 표시가 있는 기사만 크롤링하나요?**
  	>   
	> A1) 모든 기사를 크롤링하기 하기 위해서는 수 십개에 달하는 언론사의 html 구조를 모두 알아야 합니다. 따라서 일괄적으로 html 구조를 이루는 네이버뉴스 페이지로부터 크롤링을 진행합니다.
	>
    >   
	> **Q2) 어차피 "네이버뉴스" 페이지에서 크롤링할거면 애초부터 네이버 뉴스 사이트를 크롤링하면 되지 않나요?**
  	>   
	> A1) 네이버뉴스 페이지에서 검색어를 검색하면 자동으로 메인페이지에서 검색을 진행합니다.   
	> A2) 네이버금융의 뉴스·공시 탭과 마찬가지로 검색 페이지는 400페이지를 넘어가면 기사 업데이트를 해주지 않습니다. 따라서 검색 기간 설정에 제한이 걸립니다.

