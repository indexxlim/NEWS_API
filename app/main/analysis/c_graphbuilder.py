from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import compress
import copy
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager
import pandas as pd
from networkx.readwrite import json_graph
import community
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import multiprocessing
from multiprocessing import Pool
import scipy.sparse as sp


class Centrality:
    def __init__(self, input_g):
        """
        중심성을 산출하는 클래스입니다.
        :param input_g: nx graph
        """
        self.input_g = input_g

    def return_weighted_degree_centrality(self):
        w_d_centrality = {n: 0.0 for n in self.input_g.nodes()}
        for u, v, d in self.input_g.edges(data=True):
            w_d_centrality[u] += d['weight']
            w_d_centrality[v] += d['weight']
        else:
            return w_d_centrality

    def return_closeness_centrality(self):
        new_g_with_distance = self.input_g.copy()
        for u, v, d in new_g_with_distance.edges(data=True):
            d['weight'] = 1.0 / d['weight']
        return self.closeness_centrality_dev(new_g_with_distance)

    def return_betweenness_centrality(self):
        return nx.betweenness_centrality(self.input_g, weight='weight')

    def return_pagerank(self):
        return nx.pagerank(self.input_g, weight='weight')

    def return_eigenvector_centrality(self):
        return nx.eigenvector_centrality_numpy(self.input_g, weight='weight')

    def closeness_centrality_dev(self, G):  # nx.closeness_centrality 직접 수정
        A = nx.adjacency_matrix(G).tolil()
        D = scipy.sparse.csgraph.floyd_warshall( \
            A, directed=False, unweighted=False)
        n = D.shape[0]
        closeness_centrality = {}
        for r in range(0, n):
            cc = 0.0

            possible_paths = list(enumerate(D[r, :]))
            shortest_paths = dict(filter( \
                lambda x: not x[1] == np.inf, possible_paths))

            total = sum(shortest_paths.values())
            n_shortest_paths = len(shortest_paths) - 1.0
            if total > 0.0 and n > 1:
                s = n_shortest_paths / (n - 1)
                cc = (n_shortest_paths / total) * s
            closeness_centrality[r] = cc
        return closeness_centrality


class MeaningGraph:
    def __init__(self, articles, sort_by='closeness', most_freq=3, use_vn=True):
        """
        의미 네트워크를 구축합니다.
        :param articles: 데이터
        :param sentdict_path: 감성사전 경로
        :param sort_by: 그래프를 구축할 기준을 선택합니다.
            - frequency : 언급 빈도수로 정렬.
            - pagerank : pagerank로 중심성이 높은 노드부터 내림차순으로 정렬
            - betwenness : betwennes로 중심성이 높은 노드부터 내림차순으로 정렬
            - weighted_degree : weighted degree로 중심성이 높은 노드부터 내림차순으로 정렬
            - closeness : closeness로 중심성이 높은 노드부터 내림차순으로 정렬
            - eigenvector : eigenvector로 중심성이 높은 노드부터 내림차순으로 정렬
        :param use_vn: 용언과 체언 이용 여부
            - both : 용언, 체언 모두 이용합니다.
            - verb : 용언만 이용합니다.
            - noun : 체언만 이용합니다.
        """
        self.articles = articles
        self.most_freq = most_freq  # 보여줄 유사한 결과의 수 (if 3 : xx와 가장 연관의 높은 키워드 3개를 보여줌)
        self.sort_by = sort_by
        self._idx2label = {}  # co-occurrence matrix의 index와 단어를 매칭하는 dictionary
        self.word_frequency = []  # co-occurrence matrix의 index와 단어의 frequency를 매칭
        # self.vectorizer = CountVectorizer()   #to make matrix
        # self.count = self.vectorizer.fit(articles)

    def build_graph(self, min_count=50, modularity=0.05, top_n=20):
        """
        :param min_count: co-occurrence graph를 구축할 때 이용할 단어의 최소 등장 횟수
            frequency 기반인 경우에만 이용합니다.
            frequency 외의 기준에서는 기본 50으로 설정합니다.
            기본 50으로 설정한 이유는 _build_co_occurrence_matrix의 설명을 확인하세요
        :param top_n: 의미 네트워크를 구축할 때 이용할 노드의 수
            frequency가 아닌 중심성을 기준으로 그래프를 구축할 때 이용합니다.
            중심성 점수 기준 상위 top_n만큼의 노드만 남깁니다.
        :return: 인스턴스 변수 state에 그래프 정보를 저장합니다.
        """
        # 조건에 맞게 co-occurrence matrix를 생성
        self._build_co_occurrence_matrix(min_count)

        # graph를 생성하고 정렬
        G = nx.from_scipy_sparse_matrix(self.co_mat)
        score = self._calc_centrality(G, self.sort_by)

        # frequency가 아닌 중심성을 기준으로 그래프를 구축하는 경우 top_n개의 노드만 남기기
        if self.sort_by != 'frequency':
            remove_idx = list(score.keys())[top_n:]
            G.remove_nodes_from(remove_idx)
            # 연관 단어 저장
            # 제거된 노드에 맞게 index 재 정렬
            self._idx2label = {i: w for i, w in self._idx2label.items() if i in list(G.nodes)}
            self.word_frequency = self.word_frequency[list(G.nodes)]

        # edge 두께 설정
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        # 엣지 두께의 차이가 너무 커 min-max scaling에서 max 대신 75 percentile 이용
        tmp_max = np.percentile(weights, 75)
        weights_normalized = (weights - np.min(weights)) / (tmp_max - np.min(weights) + 1e-9) + 0.2  # 0.2:보정값

        # 검색 기록에 따라 연관 단어 index 변화
        # ex) 첫 검색 : 남양유업 => 남양유업을 제외하고 1번 index부터 연관 검색어 지정
        # ex) 두번째 검색 : 소비자 => 남양유업, 소비자를 제외하고 2번 index부터 연관 검색어 지정
        # hist_length = len(self._search_history)
        # self.most_freq = most_freq

        H = nx.relabel_nodes(G, self._idx2label)

        wf = 0
        for node in H.nodes():
            H.nodes[node]['weight'] = self.word_frequency[wf]
            wf = wf + 1

        v = weights_normalized

        nor_weights = (v - v.min()) / (v.max() - v.min())
        wf = 0
        for node in H.edges():
            H.edges[node]['nor_weight'] = nor_weights[wf]
            wf = wf + 1

        # modularity by threshold
        H2 = self.drop_low_weighted_edge(H, modularity)
        group = community.best_partition(H2)
        wf = 0
        for node in H2.nodes():
            H2.nodes[node]['group'] = group[node]
            wf = wf + 1
        wf = 0
        for node in H2.edges():
            H2.edges[node]['group'] = group[node[0]]
            wf = wf + 1

        data = json_graph.node_link_data(H2)
        data_str = str(data).replace("'", '"')
        data_str = str(data_str).replace("False", 'false')
        data_str = str(data_str).replace("True", 'true')

        return data_str

    def parallelize_dataframe(self, data, func):
        a = np.array_split(data, multiprocessing.cpu_count() - 1)
        pool = Pool(multiprocessing.cpu_count())
        # df = pd.concat(pool.map(func, [a,b,c,d,e]))
        data = sp.vstack(pool.map(func, a), format='csr')
        pool.close()
        pool.join()
        return data

    def test_func(self, data):
        # print("Process working on: ",data)
        tfidf_matrix = self.count.transform(data)
        # return pd.DataFrame(tfidf_matrix.toarray())
        return tfidf_matrix

    def _build_co_occurrence_matrix(self, min_count):
        """
        co-occurrence matrix를 생성합니다.
        모든 그래프는 co-occurrence matrix를 기준으로 구축됩니다.

        0) 검색어를 입력 시 검색어와 동시 등장한 단어들로 co-occurrence matrix를 구축합니다.

        1) frequency 기준일 경우 co-occurrence matrix를 기반으로
            검색어와 동시 등장 횟수가 높은 순서대로 연관 검색어를 출력합니다.
            추후 build_graph 함수로 그래프를 구축합니다.

        2) frequency 이외의 중심성 점수 기반 그래프를 구축시
            구축된 co-occurrence matrix를 기반으로 build_graph함수에서 바로 그래프를 구축합니다.
            그래프를 구축한 뒤 build_graph 함수 내에서 중심성 점수를 산출하고 상위 키워드만 남깁니다.

            이 때 최소 등장 횟수(min_count)를 0으로 설정하고 모든 단어에 대하여 그래프를 구축하면
            시간이 오래 걸리고 그래프 구축에 걸리는 시간 또한 기하급수적으로 증가합니다.
            따라서 기본 50으로 설정합니다.
            (검색 기간이 길 경우(= 출력해야할 단어가 많은 경우)에는 50으로 설정해도 오래 걸립니다,)
        """
        # CountVectorizer()를 이용한 term-document matrix 생성
        print('make graph')
        print(self.articles)

        vectorizer = CountVectorizer(min_df=0.02,max_features = 1000)
        # CountVectorizer의 기본 설정: 두글자 이상 단어만 이용
        # 한글자 단어도 포함시킬 경우 윗줄 대신 아래 코드를 이용해 vecotrizer를 정의하세요
        # vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

        X = vectorizer.fit_transform(self.articles)
        # X = self.parallelize_dataframe(articles, self.test_func)
        # word2idx = dict(sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1]))

        # index를 바탕으로 단어를 정렬합니다. 추후 그래프의 node index와 단어를 매칭하는데 이용합니다.
        word2idx = dict(sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]))

        # co-occurrence matrix 생성 / 단어 빈도수 저장
        X[X > 0] = 1  # 한 기사에 단어가 여러번 등장해도 한 번으로 설정(scaling)
        co_mat = X.T.dot(X)
        word_freq = co_mat.diagonal()
        min_max_freq = (word_freq - np.min(word_freq)) / (np.max(word_freq) - np.min(word_freq))  # min-max normalize
        word2freq = {w: min_max_freq[i] for w, i in word2idx.items()}
        co_mat.setdiag(0)

        num_art = [len(i) for i in self.articles]
        min_count = sum(num_art) / 10000

        print(min_count)

        # 최소 등장 횟수 아래로 등장한 단어를 제외하고 co-occurrence matrix를 새롭게 생성
        # 상위 키워드로 정렬하는 경우 min_count를 50으로 설정
        # if self.sort_by != 'frequency':
        #    min_count = 50  # 최소 min_count (0으로 설정할 시 너무 오래 걸리는 issue 존재)

        co_mat = co_mat.multiply(co_mat > min_count)
        nonzero_idx = co_mat.getnnz(0) > 0  # 영벡터를 제외하고 공기행렬을 새롭게 정의
        self.co_mat = co_mat[nonzero_idx][:, nonzero_idx]
        label = zip(range(len(nonzero_idx)), list(compress(word2idx.keys(), nonzero_idx)))
        self._idx2label = {i: w for i, w in label}
        # co-occurrence matrix의 index 별 word frequency
        self.word_frequency = np.array([word2freq[w] for w in self._idx2label.values()])

    def drop_low_weighted_edge(self, inputG, above_weight=0.1):
        rG = nx.Graph()
        rG.add_nodes_from(inputG.nodes(data=True))
        edges = filter(lambda e: True if e[2]['nor_weight'] >= above_weight else False, inputG.edges(data=True))
        rG.add_edges_from(edges)

        # Delete isolated node를 모두 지운다.
        for n in inputG.nodes():
            if len(list(nx.all_neighbors(rG, n))) == 0:
                rG.remove_node(n)
            # print(n, list(nx.all_neighbors(rG, n)))
        return rG

    def _calc_centrality(self, input_g, sort_by):
        """
        상위 노드를 정렬합니다.
        :param by: 정렬을 원하는 방식
            - frequency : 언급 빈도수로 정렬. co-occurrence matrix가 이미 빈도수로 정렬되어있어 별도의 처리x
            - pagerank : pagerank로 중심성이 높은 노드부터 내림차순으로 정렬
            - betwenness : betwennes로 중심성이 높은 노드부터 내림차순으로 정렬
            - weighted_degree : weighted degree로 중심성이 높은 노드부터 내림차순으로 정렬
            - closeness : closeness로 중심성이 높은 노드부터 내림차순으로 정렬
            - eigenvector : eigenvector로 중심성이 높은 노드부터 내림차순으로 정렬
        :return: centrality score of every node
        """

        cent = Centrality(input_g)
        if sort_by == 'frequency':
            score = None
        elif sort_by == 'pagerank':
            score = cent.return_pagerank()
        elif sort_by == 'betwenness':
            score = cent.return_betweenness_centrality()
        elif sort_by == 'weighted_degree':
            score = cent.return_weighted_degree_centrality()
        elif sort_by == 'closeness':
            score = cent.return_closeness_centrality()
        elif sort_by == 'eigenvector':
            score = cent.return_eigenvector_centrality()
        else:
            raise ValueError("잘못된 기준값입니다.")

        if score is not None:
            score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))

        return score


def tfidftable(bb):
    cv = CountVectorizer()  # max_features 수정
    tdm = cv.fit_transform(bb)

    ##TF-IDF
    tfidf = TfidfTransformer()
    tdmtfidf = tfidf.fit_transfowrm(tdm)
    words = cv.get_feature_names()  # 단어 추출

    # sum tfidf frequency of each term through documents
    sums = tdmtfidf.sum(axis=0)

    # connecting term to its sums frequency
    data = []
    for col, term in enumerate(words):
        data.append((term, sums[0, col]))

    tfidftable = pd.DataFrame(data, columns=['키워드', 'TF-IDF'])
    tfidftable = tfidftable.set_index('키워드')
    tfidftable = tfidftable.sort_values('TF-IDF', ascending=False)
    return tfidftable.iloc[0:100].to_json(force_ascii=False)

#
# if __name__ == ""__main__"":
#     path1 = 'D:/업무/조국/언론사별_조사_총합/'
#     bb = pd.read_csv(path1+'중앙일보_article.csv', encoding='ansi')
#
#     keyword = '의혹'
#     articles = bb[list(map(lambda x: keyword in x, bb['본문내용']))]
#     articles = articles['본문내용']
#
#     print(graph2json(articles, keyword))