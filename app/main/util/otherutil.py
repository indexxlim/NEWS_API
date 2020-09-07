import pandas as pd
import urllib.parse
import re
from konlpy.tag import Mecab
import gensim

bannedwords = ['출처', '기자', '하지', '하기', '천억', '진짜', '이것', '저것', '누구', '무엇', '그것', '위해', '역시', '이런', '저런', '그런', '무슨',
               '누가', '요즘', '끼리', '가지', '정말', '보기', '하나', '이제', '어디', '라이', '아주', '바로', '자기', '그냥', '지금',
               '바로', '그냥', '다른', '이번', '해주', '언제', '때문', '완전', '이건', '보고', '얼마나', '모두', '너희', '우리', '당신', '해도',
               '해주',
               '하라', '건가', '요게', '그게', '이게', '그거', '저거', '지랄', '대가리', '이날', '오전', '관련', '동양', '입장', '오후', '당시',
               '가능',
               '과정', '경우', '생각', '진행', '이상', '시작', '가운데', '정도', '이후', '내용', '주장', '확인', '공개', '일부', '준비', '대상',
               '부분', '핵심',
               '상태', '입시', '보도', '결과', '동안', '시절', '차례', '다음', '포함', '학원', '사모', '정보', '언급', '지난', '지난달', '사이',
               '왼쪽', '처음', '상대',
               '결국', '직후', '개인', '얘기', '대신', '여기', '사실', '최근', '전날', '뉴스', '사람', '전문', '본인', '제기', '있다',
               '있다.', '무단', '배포', '니다']
def build_csv_data(dataframe):
    csv_data = dataframe.to_csv(index=True, encoding='utf-8')
    csv_data = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_data)
    return csv_data

def preprocess(article_list, split=True):


    mecab = Mecab()

    ##text 전체 명사화
    a = []
    for i in range(len(article_list)):
        try:
            b = mecab.nouns(article_list[i])
            somelist = [x for x in b if x not in bannedwords]
            a.append(somelist)
        except:
            pass

    if split:
        ##문장으로 변환
        bb = [' '.join(a[0])]
        for i in range(1, len(a)):
            bb = bb + [' '.join(a[i])]
    else:
        return a

    return bb


def boot_table(df):
    table = df.to_html(classes='table', header=True, index=False)

    table = re.sub(
        r'<table([^>]*)>',
        r'<table\1 id="table_sortSearchResult"'
        r'name="table"'
        r'data-toggle="table"'
        r'data-maintain-selected = "true"'
        r'data-sort-name = "pdate"'
        r'data-sort-order = "asc"'
        r'data-search = "true"'
        r'data-show-pagination-switch = "true"'
        r'data-pagination = "true"'
        r'data-page-list = "[10, 25, 50, 100, ALL]"'
        r'data-page-size = "25"'
        r'data-show-footer = "false"'
        r'data-side-pagination = "client"'
        r'data-show-export = "true"'
        r'data-export-types = "'+ "['excel']"+ '"'
        # r'data-export-options = "{    }"'
        r'data-click-to-select = "true"> ',

        table
    )

    table = [table]

    return table