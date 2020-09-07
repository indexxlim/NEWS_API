import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import time, random
import pandas as pd
import pprint
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from multiprocessing import Pool
from datetime import date


def get_news_daum(n_url):
    print('news_url : ', n_url)
    breq = requests.get(n_url)
    print(breq)
    bsoup = BeautifulSoup(breq.content, 'html.parser')
    
    # 날짜 파싱
    pdate = bsoup.select('.txt_info')[-1].get_text().replace(" ",'')[2:12]
    # 기사 제목
    title = bsoup.select('title')[0].get_text().replace(" | Daum 뉴스","")
    # 기사 본문 크롤링
    btext = bsoup.select('div.news_view')[0].get_text().replace("\n","").strip()
    # 신문사 크롤링
    try:
        pcompany = bsoup.find_all('meta',{'name':'article:media_name'})[0].get('content')
    except:
        pcompany = ''
    #url
    #분류명
    aclass = bsoup.select('h2#kakaoBody')[0].get_text()
    
    #s = requests.session()
    #s.config['keep_alive'] = False
    
    #'pdate', 'articleTitle', 'article', 'pcompany', 'url'
    return pdate, title, btext, pcompany, n_url, aclass

def get_news_daum_df(n_url):
    print('news_url : ', n_url)
    breq = requests.get(n_url)
    bsoup = BeautifulSoup(breq.content, 'html.parser')
    
    # 날짜 파싱
    pdate = bsoup.select('.txt_info')[-1].get_text().replace(" ",'')[2:12]
    # 기사 제목
    title = bsoup.select('title')[0].get_text().replace(" | Daum 뉴스","")
    # 기사 본문 크롤링
    btext = bsoup.select('div.news_view')[0].get_text().replace("\n","").strip()
    # 신문사 크롤링
    try:
        pcompany = bsoup.find_all('meta',{'name':'article:media_name'})[0].get('content')
    except:
        pcompany = ''
    #url
    #분류명
    aclass = bsoup.select('h2#kakaoBody')[0].get_text()
    
    #'pdate', 'articleTitle', 'article', 'pcompany', 'url'
    return pd.DataFrame([[pdate, title, btext, pcompany, n_url, aclass]],
                        columns=['pdate', 'articleTitle', 'article', 'pcompany', 'url', 'aclass'])
                        
                        
def get_daum_comment(n_url):
    columns = ['content', 'createdAt', 'likeCount', 'dislikeCount']

    comment_df = pd.DataFrame(columns = columns)
    list_one = get_news_daum(n_url)

    req = requests.get(n_url)
    cont = req.content
    soup = BeautifulSoup(cont, 'html.parser')

    area = soup.find_all("div", "alex-area")[0]
    client_id = area['data-client-id']
     
    url2 = "https://comment.daum.net/oauth/token?grant_type=alex_credentials&client_id="+client_id
    req = requests.get(url2, headers={"Referer": n_url})
    auth = req.text

    header2 = { "Authorization":"Bearer "+json.loads(auth)['access_token']}
    post_id = n_url.split('/')[-1]
    turl = "http://comment.daum.net/apis/v1/posts/@"+ post_id
    tar = requests.get(turl, headers = header2)

    limit = json.loads(tar.text)['commentCount']
    commentId = json.loads(tar.text)['id']
    parentId = 0
    sort='favorite'
    durl = "http://comment.daum.net/apis/v1/posts/"+str(commentId)+"/comments?parentId="+str(parentId)#+"&offset=0&limit="+str(limit)+"&sort="+sort
    dat = requests.get(durl)
    dat = json.loads(dat.text)
    if len(dat) == 0:
        return comment_df



    for i in range(int(limit/100)+1):
        durl = "http://comment.daum.net/apis/v1/posts/"+str(commentId)+"/comments?parentId="+str(parentId)+"&offset=" + str(i*100) +  "&limit="+"100"+"&sort="+sort    
        dat = requests.get(durl)
        dat = json.loads(dat.text)
   
        try:
            data = json_normalize(dat)[['content','createdAt', 'likeCount', 'dislikeCount']]
        except:
            print('maximum number of news is ', i*100)
            break
        
        comment_df = comment_df.append(data)


    
    columns = ['pdate', 'articleTitle', 'article', 'pcompany', 'url', 'aclass' ,'comment', 'sympathyCount', 'antipathyCount']
    news_comment_df = pd.DataFrame(columns = columns)


    
    n_con = len(data['content'])
    data2 = [[list_one[0]]*n_con, [list_one[1]]*n_con, [list_one[2]]*n_con, [list_one[3]]*n_con, [list_one[4]]*n_con, [list_one[5]]*n_con, list(data['content']), list(data['likeCount']), list(data['dislikeCount'])]
    list(map(list, zip(*data2)))
    news_comment_df = pd.DataFrame(list(map(list, zip(*data2))),columns=columns)


    return comment_df
    
def get_rank_news_daum(cdate = date.today()):
    cdate = str(cdate).replace(".","")
    cdate = str(cdate).replace("-", '')
    
    url = 'https://media.daum.net/ranking/bestreply?regDate='+str(int(cdate)-1)
    req = requests.get(url)

    cont = req.text
    soup = BeautifulSoup(cont, 'html.parser')
    href_list = [i['href'] for i in soup.select('a.link_thumb')]
    return href_list



def func(x):
    if 'daum.net' in x:
        return x
    else:
        return None
        
def search_daum_news(query, s_date, e_date):
    s_from = s_date.replace(".","")
    e_to = e_date.replace(".","")
    
    dt_index = pd.date_range(start=s_from, end=e_to)
    dt_list = dt_index.strftime("%Y%m%d").tolist()
    
    df = pd.DataFrame(columns = ['pdate', 'articleTitle', 'article', 'pcompany', 'url', 'aclass'])

    for i in dt_list:        
        print(i)
        df_one = search_daum_news_alldays(query, i, i)
        if type(df_one) == pd.core.frame.DataFrame:
            df = df.append(df_one,ignore_index=True)

    return df
            

def search_daum_news_alldays(query, s_date, e_date):
    s_date = str(s_date).replace(".","")
    e_date = str(e_date).replace(".","")
    
    pages=1000000
    news_list=[]
    daum_list=pd.DataFrame()
    for page in range(pages):
        url="https://search.daum.net/search?nil_suggest=btn&w=news&DA=PGD&cluster=y&q="+query+"&sd="+s_date+"000000&ed="+e_date+"235959&period=u&p="+str(page+1)
        print(url)
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        if len(soup.find_all("div", {"class":"result_message mg_cont"}))==1:
            break
        con = soup.find_all("a", {"class":"f_nb"})
        if len(con) == 0 :
            return []
        for i in range(len(con)):
            news_list.append(con[i]['href'])
    news_list=list(set(list(filter(func, news_list))))
    if len(news_list) == 0:
        return news_list
    with Pool(10) as p:
        get_news_list = p.map(get_news_daum_df, news_list)

    daum_list = daum_list.append(pd.concat(get_news_list))
    # for turl in news_list:
        # daum_frame = get_news_daum_df(turl)
        
        # daum_list=pd.concat([daum_list, daum_frame])
    return daum_list

    
    
def get_rank_news_daum():

    
#https://search.daum.net/search?w=news&q=%EC%9C%A0%ED%8A%9C%EB%B8%8C
    return 0