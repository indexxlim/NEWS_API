from flask import request, jsonify
from flask_restplus import Resource
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
from konlpy.tag import Mecab
import requests

import threading
from app.main.util.dto import News
from app.main.newstone.preprocess import predict, vec_predict
from app.main.analysis.prwi import prwi
from app.main.newstone.sent_analysis import sent_analysis
from app.main.analysis.graphbuilder import graph2json
from app.main.analysis.c_graphbuilder import MeaningGraph

from app.main.analysis.topicmodel import TopicModelViz
from app.main.util.otherutil import bannedwords, preprocess
from flaskthreads import AppContextThread

##celery
import task
import task_short
import task_analysis
from flask import current_app

api = News.api
mecab = Mecab()


@api.route('/sentiment')
class NewsTone(Resource):
    @api.doc('predict news tone')
    def post(self):
        """predict news tone"""

        print(threading.active_count())

        request_data = request.get_json()

        jsondf = json_normalize(request_data['news_data'])
        jsondf = jsondf.fillna('')
        key = jsondf['key']
        # total_pred, message, status = predict(jsondf['article'])  #predict
        total_pred, message, status, pos, nuet, neg  = task.sentiment(list(jsondf['article']),500)
        current_app.logger.debug('Analysis Status: %s', message)
        del jsondf
        # prediction = np.array2string(total_pred)
        if status < 0:
            result = total_pred
            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        else:
            df = pd.DataFrame({'key': key, 'predict': total_pred, 'pos' : pos, 'nuet' : nuet, 'neg':neg})
            result = json.loads(df.to_json(orient='records'))

            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        return jsonify(response_object)

@api.route('/sentiment2')
class NewsTone2(Resource):
    @api.doc('predict news tone')
    def post(self):
        """predict news tone"""

        print(threading.active_count())

        request_data = request.get_json()

        jsondf = json_normalize(request_data['news_data'])
        jsondf = jsondf.fillna('')
        key = jsondf['key']
        # total_pred, message, status = predict(jsondf['article'])  #predict
        total_pred, message, status, pos, nuet, neg  = task.sentiment2(list(jsondf['article']))
        del jsondf
        # prediction = np.array2string(total_pred)
        if status < 0:
            result = total_pred
            json_normalize
            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        else:
            df = pd.DataFrame({'key': key, 'predict': total_pred, 'pos' : pos, 'nuet' : nuet, 'neg':neg})
            result = json.loads(df.to_json(orient='records'))

            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        return jsonify(response_object)


@api.route('/short')
class NewsTone2(Resource):
    @api.doc('predict news tone')
    def post(self):
        """predict news tone"""

        print(threading.active_count())

        request_data = request.get_json()

        jsondf = json_normalize(request_data['news_data'])
        jsondf = jsondf.fillna('')
        key = jsondf['key']
        # total_pred, message, status = predict(jsondf['article'])  #predict
        total_pred, message, status, pos, nuet, neg = task_short.inference(list(jsondf['article']))
        del jsondf
        # prediction = np.array2string(total_pred)
        if status < 0:
            result = total_pred
            json_normalize
            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        else:
            df = pd.DataFrame({'key': key, 'predict': total_pred, 'pos' : pos, 'nuet' : nuet, 'neg':neg})
            result = json.loads(df.to_json(orient='records'))

            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        return jsonify(response_object)


@api.route('/short2')
class NewsTone2(Resource):
    @api.doc('predict news tone')
    def post(self):
        """predict news tone"""

        print(threading.active_count())

        request_data = request.get_json()

        jsondf = json_normalize(request_data['news_data'])
        jsondf = jsondf.fillna('')
        key = jsondf['key']
        # total_pred, message, status = predict(jsondf['article'])  #predict
        total_pred, message, status = task_short.short2(list(jsondf['article']))
        del jsondf
        # prediction = np.array2string(total_pred)
        if status < 0:
            result = total_pred
            json_normalize
            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        else:
            df = pd.DataFrame({'key': key, 'predict': total_pred})
            result = json.loads(df.to_json(orient='records'))

            response_object = {
                'status': status,
                'message': message,
                'result': result
            }
        return jsonify(response_object)


@api.route('/specific')
class NewsTone(Resource):
    @api.doc('predict news tone')
    def post(self):
        """predict news tone"""

        request_data = request.get_json()

        jsondf = json_normalize(request_data['news_data'])
        query = request_data['query']
        synonym = request_data['synonym']

        key = jsondf['key']
        # print(jsondf['title'])
        # print(jsondf['article'])
        total_pred, status = prwi(query, jsondf['title'], jsondf['article'], synonym)  # predict

        del jsondf

        # prediction = np.array2string(total_pred)
        df = pd.DataFrame({'key': key, 'predict': total_pred})
        result = json.loads(df.to_json(orient='records'))

        response_object = {
            'status': status,
            'message': 'Ok',
            'result': result
        }

        return jsonify(response_object)


@api.route('/sentdic')
class NewsTone(Resource):
    @api.doc('predict word news tone')
    def post(self):

        request_data = request.get_json()

        jsondf = json_normalize(request_data['news_data'])
        key = jsondf['key']
        # print(jsondf['title'])
        # print(jsondf['article'])
        wordsenti_score, status = sent_analysis(jsondf['article'])  # predict

        del jsondf

        if status < 0:
            result = wordsenti_score
        else:
            # prediction = np.array2string(total_pred)
            df = pd.DataFrame({'key': key, 'predict': wordsenti_score})
            result = json.loads(df.to_json(orient='records'))

        response_object = {
            'status': status,
            'message': 'Ok',
            'result': result
        }

        return jsonify(response_object)


def send_sna(url, key, article, top_n):
#    task_analysis.sna.delay(url, key, article,top_n).get()
    task_analysis.sna(url, key, article,top_n)

    #executor.submit(task_analysis.sna, url, key, article,top_n)



    #
    # try:
    #     mg = MeaningGraph(article)
    #     graph = mg.build_graph(None)
    #     # graph = graph2json(article)
    #
    # except Exception as inst:
    #     graph = 'modelling error'
    #     print(type(inst))
    #     print(inst.args)
    #     print(inst)
    #
    #
    # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    # request_data = {
    #     "id": key,
    #     "message": graph
    # }
    # j_data = json.dumps(request_data)
    #
    #
    # print(j_data)
    #
    # r = requests.post(url, data=j_data, headers=headers)
    # print(r)

@api.route('/SNA')
class NewsTone(Resource):
    @api.doc('predict news tone')
    def post(self):



        request_data = request.get_json()

        # jsondf = json_normalize(request_data['news_data'])
        # keyword = request_data['keyword']
        # article = np.array(jsondf['article'])

        # key = jsondf['key']
        # print(jsondf['title'])
        bannedwords
        article = request_data['contents'].split('\n')
        url = request_data['url']

        key = request_data['id']
        if 'count' in request_data:
            top_n= request_data['count']
        else:
            top_n = 50
        # if len(article)<10:
        #     response_object = {
        #         'status': 1,
        #         'message': 'graph input data Should be large(over 50)',
        #         'result': []
        #     }
        ##text 전체 명사화
        # else:
        # news_list = preprocess(article)
        #        send_sna(url, key, article)

        t = AppContextThread(target=send_sna, args=(url, key, article,top_n))
        #executor.submit(task_analysis.sna, url, key, article, top_n)

        t.start()

        try:
            response_object = {
                'status': 1,
                'message': 'Ok',
                'result': 'redirect to url'
            }
        except:
            response_object = {
                'status': 1,
                'message': 'graph error',
                'result': 'redirect to url'
            }

        return response_object


def send_topic(url, key, article):
    task_analysis.topic.delay(url, key, article).get()

    # news_list = preprocess(article, split=False)
    # tmv = TopicModelViz()
    #
    # try:
    #     print('topic')
    #     dic, corpus = tmv.lda_preprocess(article)
    #     lda = tmv.lda_model(dic, corpus,
    #                         num_topics=5,
    #                         eval_every=5)  # predict
    #
    #     lda_vis = pyLDAvis.gensim.prepare(lda, corpus, dic)
    #     message = lda_vis.to_json()
    #
    # except:
    #     message = 'modelling error'
    #     print(message)
    #
    # print(message)
    # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    # request_data = {
    #     "id": key,
    #     "message": message
    #
    # }
    # j_data = json.dumps(request_data)
    # print(j_data)
    #
    # r = requests.post(url, data=j_data, headers=headers)
    # return (r)


@api.route('/topic')
class NewsTone(Resource):
    @api.doc('predict news tone')
    def post(self):

        request_data = request.get_json()

        # jsondf = json_normalize(request_data['news_data'])
        # #keyword = request_data['keyword']
        # article = np.array(jsondf['article'])
        # key = jsondf['key']


        article = request_data['contents'].split('\n')
        url = request_data['url']
        key = request_data['id']
        # print(jsondf['title'])
        if len(article) < 10:
            response_object = {
                'status': 1,
                'message': 'graph input data Should be large(over 50)',
                'result': []
            }
        else:
            ##text 전체 명사화
            # send_topic(url, key, article)
            t = AppContextThread(target=send_topic, args=(url, key, article))
            t.start()

            # news_list = preprocess(article, split=False)
            # tmv = TopicModelViz()

            # try:
            #     dic, corpus = tmv.lda_preprocess(news_list)
            #     lda = tmv.lda_model(dic, corpus,
            #                         num_topics=5,
            #                         eval_every=5)  # predict
            #
            #     lda_vis = pyLDAvis.gensim.prepare(lda, corpus, dic)
            #     response_object = {
            #         'status': 1,
            #         'message': 'Ok',
            #         'result': lda_vis.to_json()
            #     }
            # except:
            #     response_object = {
            #         'status': 1,
            #         'message': 'modelling error',
            #         'result': []
            #     }
            response_object = {
                'status': 1,
                'message': 'Ok',
                'result': 'redirect to url'
            }
        return response_object


@api.route('/vector')
class NewsTone(Resource):
    @api.doc('predict news tone')
    def post(self):
        """predict news tone"""

        request_data = request.get_json()

        total_pred = vec_predict(request_data['news_data'])

        prediction = np.array2string(total_pred)

        return jsonify(prediction)

    # return 'Hello, World!'



