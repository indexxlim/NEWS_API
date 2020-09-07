import torch
import time
from celery import Celery
import numpy as np
import json
from konlpy.tag import Mecab


from app.main.analysis.graphbuilder import graph2json
from app.main.analysis.c_graphbuilder import MeaningGraph
from app.main.analysis.topicmodel import TopicModelViz
import pyLDAvis


import requests
import time

app = Celery('task', backend='amqp', broker='amqp://guest@localhost:5672')

print('task_analysis synchronization')
mecab = Mecab()

app.conf.update(
    CELERY_TIMEZONE='Asia/Seoul',
    WORKER_MAX_MEMORY_PER_CHILD=20000000
)


@app.task
def sna(url, key, article, top_n):
    try:
        graph = graph2json(article, top_n=top_n)
        # mg = MeaningGraph(article)
        # graph = mg.build_graph()
        # print(graph)
    except Exception as inst:
        graph = 'modelling error'
        print(type(inst))
        print(inst.args)
        print(inst)

    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    request_data = {
        "id": key,
        "message": graph
    }
    j_data = json.dumps(request_data)

    #url = 'http://192.168.0.48:5001/query'
    r = requests.post(url, data=j_data, headers=headers)
    print(r)

@app.task
def topic(url, key, article):
    # news_list = preprocess(article, split=False)
    tmv = TopicModelViz()

    start_time = time.time()

    try:
        print('topic')
        dic, corpus = tmv.lda_preprocess(article)
        lda = tmv.lda_model(dic, corpus,
                            num_topics=5,
                            eval_every=5)  # predict

        lda_vis = pyLDAvis.gensim.prepare(lda, corpus, dic)
        message = lda_vis.to_json()

    except Exception as inst:
        message = 'modelling error'
        print(type(inst))
        print(inst.args)
        print(inst)


    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    request_data = {
        "id": key,
        "message": message

    }
    j_data = json.dumps(request_data)


    end_time = time.time()
    print("WorkingTime: {} sec".format(end_time - start_time))

    r = requests.post(url, data=j_data, headers=headers)
    print(r)
