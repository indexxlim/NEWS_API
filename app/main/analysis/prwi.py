import numpy as np
import regex
from konlpy.tag import Mecab
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tnrange, tqdm
import math
from sklearn.metrics import *
from app.main.newstone.word2vec import w2v
from ..newstone.utils import clean_text


mecab=Mecab()

##보도 비중 계수 함수 선언(키워드, 제목, 본문, 유사어)
def prwi_cal(query, title, news, *synonym):
    _wv = w2v.word2vec

    total_title_sim = []
    total_news_sim = []
    total_query_density = []
    news = clean_text(news)

    for k in query.split():
        pos_query = [f'{word}_{pos}' for word, pos in mecab.pos(k)]
        if len(pos_query) > 1:
            total_query_vector = np.zeros(100)
            for query_morph in pos_query:
                query_morph_vector = _wv[query_morph]
                total_query_vector = total_query_vector + query_morph_vector
            query_vectors = total_query_vector / len(pos_query)
        else:
            query_vectors = _wv[pos_query]

        pos_news = [f'{word}_{pos}' for word, pos in mecab.pos(news)]
        pos_news = [word for word in pos_news if 'N' in word or 'NR' in word or 'NP' in word]

        if len(synonym) != 0:
            pos_synonym = [f'{word}_{pos}' for word, pos in mecab.pos(''.join(''.join(synonym).split(',')))]
            pos_news = [''.join(pos_query) if x in pos_synonym else x for x in pos_news]
        else:
            pass

        pos_title = [f'{word}_{pos}' for word, pos in mecab.pos(title)]
        pos_title = [word for word in pos_title if 'N' in word or 'NR' in word or 'NP' in word]

        news_vectors_lead = np.zeros(100)
        news_vectors_else = np.zeros(100)
        lead_query_freq = 0
        for i in range(len(pos_news)):
            if i < math.floor(len(pos_news) * 0.1):
                try:
                    news_tok_vectors_lead = _wv[pos_news[i]]
                except:
                    news_tok_vectors_lead = np.zeros(100)
                news_vectors_lead = news_vectors_lead + news_tok_vectors_lead
                if ''.join(pos_query) == pos_news[i]:
                    lead_query_freq = lead_query_freq + 1
                else:
                    pass
            else:
                try:
                    news_tok_vectors_else = _wv[pos_news[i]]
                except:
                    news_tok_vectors_else = np.zeros(100)
                news_vectors_else = news_vectors_else + news_tok_vectors_else

        if len(pos_news) > 0:
            news_vectors_avg_lead = news_vectors_lead / math.ceil(len(pos_news) * 0.1)
            news_sim_lead = float(np.dot(query_vectors, news_vectors_avg_lead) / (
                        np.linalg.norm(query_vectors) * np.linalg.norm(news_vectors_avg_lead)))
            if np.isnan(news_sim_lead) == True:
                news_sim_lead = 0
            else:
                pass
            news_vectors_avg_else = news_vectors_else / (len(pos_news) - math.ceil(len(pos_news) * 0.1))
            news_sim_else = float(np.dot(query_vectors, news_vectors_avg_else) / (
                        np.linalg.norm(query_vectors) * np.linalg.norm(news_vectors_avg_else)))
            if np.isnan(news_sim_else) == True:
                news_sim_else = 0
            else:
                pass
            news_sim = float(((news_sim_lead * 1.3) + (news_sim_else * 0.7)) / 2)
        else:
            news_sim = float(0)

        if lead_query_freq > 0:
            news_sim = news_sim * ((lead_query_freq * 0.1) + 1.2)
        else:
            pass

        total_news_sim.append(news_sim)

        title_vectors = np.zeros(100)
        for i in range(len(pos_title)):
            try:
                title_tok_vectors = _wv[pos_title[i]]
            except:
                title_tok_vectors = np.zeros(100)
            title_vectors = title_vectors + title_tok_vectors
        if len(pos_title) > 0:
            title_vectors_avg = title_vectors / len(pos_title)
            title_sim = float(np.dot(query_vectors, title_vectors_avg) / (
                        np.linalg.norm(query_vectors) * np.linalg.norm(title_vectors_avg)))
            if np.isnan(title_sim) == True:
                title_sim = 0
            else:
                pass
        else:
            title_sim = float(0)

        title_query_freq = 0
        for i in range(len(pos_title)):
            if ''.join(pos_query) == pos_title[i]:
                title_query_freq = title_query_freq + 1
            else:
                pass

        if title_query_freq > 0:
            title_sim = title_sim * ((title_query_freq * 0.1) + 1.2)
        else:
            pass
        total_title_sim.append(title_sim)

        news_query_freq = 0
        for i in range(len(pos_news)):
            if ''.join(pos_query) == pos_news[i]:
                news_query_freq = news_query_freq + 1
            else:
                pass

        if news_query_freq != 0 and len(pos_news) != 0:
            query_density = news_query_freq / len(pos_news)
        else:
            query_density = 0
        total_query_density.append(query_density)

    fin_news_sim = float(sum(total_news_sim) / len(query.split()))
    fin_title_sim = float(sum(total_title_sim) / len(query.split()))
    fin_query_density = float(sum(total_query_density) / len(query.split()))
    total_score = fin_title_sim + fin_news_sim + (5 * fin_query_density)

    ### 범주화(XXX: 비중 없음 기준 / YYY: 비중 높음 기준) - 범주화 기준 세워지기 전까지 categorization 함수 활용
    # if total_score>high_v:
    # score_category=2
    # elif total_score<high_v and total_score>low_v:
    # score_category=1
    # else:
    # score_category=0

    return total_score

## 보도 비중 계수 산출 실행 함수 선언(키워드, 제목, 본문, 유사어)
def prwi(query, title='', text='', *synonym):
    y_pred=[]
    synonym=''.join(''.join(synonym).split(','))
    try:
        for i in tqdm(range(len(text))):
            sim=prwi_cal(query, title[i], text[i], synonym)
            y_pred.append(sim)
    except:
        return 'error prwi_cal', -1
    return y_pred, 1
