# -*- coding: utf-8 -*-

import argparse
import pickle
import gensim
import numpy as np
import pandas as pd
import pyLDAvis.gensim

from tqdm import tqdm
from gensim import corpora
from gensim.models.ldamodel import LdaModel


# from gensim.models.ldamulticore import LdaMulticore


class TopicModelViz:
    """gensim을 이용한 LDA Modeling 후 pyLDAvis를 이용해 시각화하는 클래스"""

    def create_datframe(self, data, media=None,
                        from_date='2019-04-10', to_date='2019-04-13'):
        '''pickle 데이터를 DataFrame으로 만드는 함수
        Args:
            - data: pickle data [(매체명, 기사번호, 발행일, 분류명, 제목, 본문내용), ...]
        Returns:
            - df: DataFrame
        '''
        columns = ['매채명', '기사번호', '발행일', '분류명', '제목', '본문내용']

        df = pd.DataFrame(data, columns=columns)
        if not media:
            df = df.query(f"'{from_date}' <= 발행일 <= '{to_date}'")
        elif media:
            df = df.query(f"'{from_date}' <= 발행일 <= '{to_date}' and 매채명 == '{media}'")

        return df

    def lda_preprocess_df(self, data):
        """"""
        article_list = []
        for article in tqdm(data['본문내용'].tolist()):
            article = [word for word in article if len(word) > 1]
            article_list.append(article)

        dictionary = corpora.Dictionary(article_list)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in article_list]

        return dictionary, doc_term_matrix

    def lda_preprocess(self, data):
        """"""
        article_list = []
        for article in (data):
            article = article.split(' ')
            article = [word for word in article if len(word) > 1]
            article_list.append(article)

        dictionary = corpora.Dictionary(article_list)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in article_list]

        return dictionary, doc_term_matrix

    def lda_model(self, dictionary, corpus,
                  num_topics=10, eval_every=5, save_path=None):
        lda = LdaModel(corpus, num_topics=num_topics,
                       id2word=dictionary, eval_every=eval_every)

        if save_path:
            lda.save(save_path)

        return lda

    def lda_visualize(self, lda, corpus, dictionary, dir_path='./lda_vis'):
        lda_vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(lda_vis, f'{dir_path}.html')

        #########################################################################
        ##### 2019.10.09 추가
        ##### - 내용 : TopicModel 결과 DataFrame 저장
        ##### - topic_df : 각 토픽(Category)에 해당하는 단어의 빈도수를 나타낸 DF
        ##### - token_df : 각 토픽 별 단어 스코어를 나타낸 DF
        #########################################################################
        topic_df = lda_vis.topic_info[['Category', 'Term', 'Freq']]
        topic_df.to_csv(f'{dir_path}_topic_info.csv', index=False)

        token_df = lda_vis.token_table.reset_index()
        token_df = token_df[['Term', 'Topic', 'Freq']]
        token_df.columns = ['Term', 'Topic', 'Score']
        token_df.to_csv(f'{dir_path}_token_table.csv', index=False)

        return None


if __name__ == "__main__":
    ############################
    # 0. Argument 받기
    ############################
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--from_date', type=str, default='2019-04-10',
                        help="시작 날짜를 입력해 주세요 (형식 : 'y-m-d')")
    parser.add_argument('--to_date', type=str, default='2019-04-13',
                        help="종료 날짜를 입력해 주세요 (형식 : 'y-m-d')")
    parser.add_argument('--num_topics', type=int, default=10,
                        help="생성할 토픽의 개수를 입력해 주세요 (형식 : 1, 2, 3, ...)")
    parser.add_argument('--eval_every', type=int, default=5)
    args = parser.parse_args()

    #####################
    # 1. Data Load
    #####################
    print("데이터를 불러 오고 있습니다...")
    # [(매체명, 기사번호, 발행일, 분류명, 제목, 본문내용), ...]
    with open('./data/namyang_articles_nouns-v01.txt', 'rb') as fp:
        data = pickle.load(fp)

    ##############################
    # 2. LDA Model에 필요한 전처리
    #  - dictionary, corpus 생성
    ###############################
    print("LDA 모델링 중 입니다...")
    tmv = TopicModelViz()
    df = tmv.create_datframe(data, from_date=args.from_date, to_date=args.to_date)
    dic, corpus = tmv.lda_preprocess_df(df)

    ####################
    # 3. LDA Modeling
    ####################
    lda = tmv.lda_model(dic, corpus,
                        num_topics=args.num_topics,
                        eval_every=args.eval_every)

    ####################
    # 4. Save pyLDAvis
    ####################
    print("pyLDAvis 결과 파일들을 생성중 입니다. 다소 시간이 소요 됩니다...")
    dir_path = './lda_vis'
    tmv.lda_visualize(lda, corpus, dic, dir_path)

    print('완료 되었습니다.')