from flask import Flask
#from flask_sqlalchemy import SQLAlchemy
#from flask_bcrypt import Bcrypt
#import flask_monitoringdashboard as dashboard
from flask_script import Command, Option
from flask_bootstrap import Bootstrap

from .config import config_by_name
from .newstone.word2vec import w2v
from .newstone.CNN_Model import dmodel
from flask import render_template,request,redirect,url_for
import urllib.request
import pandas as pd
import numpy as np


from app.main.spidering.collect_naver_news import get_naver_comment, get_naver_comment_list
from app.main.spidering.collect_daum_news import get_daum_comment
from app.main.util.otherutil import build_csv_data, boot_table, preprocess
from app.main.analysis.graphbuilder import tfidftable
from app.main.templates.forms import *
from app.main.analysis.graphbuilder import graph2json
from multiprocessing import Pool
#from flask_executor import Executor


#executor = Executor()
#db = SQLAlchemy()
#flask_bcrypt = Bcrypt()
def news_df(result):
    # if len(result.split(';')) > 1:
    #     df = get_naver_comment_list(result.split(';'))

    if 'naver' in result:
        comment_df = get_naver_comment(result)
        comment_df=comment_df.drop(['article','url'],axis=1)
        return comment_df

    elif 'daum' in result:
        comment_df = get_daum_comment(result)
        comment_df=comment_df.drop(['article','url'],axis=1)
        return comment_df
    else:
        return []


def create_app(config_name):
    app = Flask(__name__)
#    dashboard.bind(app)
    app.config.from_object(config_by_name[config_name])
    #db.init_app(app)
    #flask_bcrypt.init_app(app)
    Bootstrap(app)
    #executor.init_app(app)
    #app.config['EXECUTOR_TYPE'] = 'thread'


    @app.route('/web')
    def web():
        return render_template('bootstrap.html')

    @app.route('/home')
    def home(message=None):
        return render_template('pages/placeholder.home.html', message=message)

    @app.route('/comment_list', methods=['GET', 'POST'])
    def comment_list():
        if request.method == 'POST':
            result = request.form

            try:
                comment_df = news_df(result['url'])
                if len(comment_df) == 0:
                    message = "올바른 URL이 아닙니다."
                    return redirect(url_for('home'))
            except:
                message = "올바른 URL이 아닙니다."
                return redirect(url_for('home'))
            table = boot_table(comment_df)

            return render_template("pages/placeholder.comment_list.html", result=result, tables=table)
        else:
            result = dict([('url', 'no url')])
            return render_template('pages/placeholder.comment_list.html', result=result)



    @app.route('/network', methods=['GET', 'POST'])
    def network():
        if request.method == 'POST':
            table = request.form['table']
            df_table = pd.read_html(table)[0]
            news_list = np.reshape(np.array(df_table['comment']), (len(df_table['comment'])))
            news_list = preprocess(news_list)
            word_table = tfidftable(news_list)
            graph = graph2json(news_list)
            print(graph)
            return render_template('pages/placeholder.network.html', graph=graph, word_table=word_table)

        # if request.method=='POST':
        #		url = request.form['url']
        else:
            with urllib.request.urlopen("https://raw.githubusercontent.com/indexxlim/Jsondata/master/data_str2.json") as url:
                s = url.read()
            graph = s.decode("utf-8")
            return render_template('pages/placeholder.network.html', graph=graph)

    @app.route('/about')
    def about():
        return render_template('pages/placeholder.about.html')

    @app.route('/login')
    def login():
        form = LoginForm(request.form)
        return render_template('forms/login.html', form=form)

    @app.route('/register')
    def register():
        form = RegisterForm(request.form)
        return render_template('forms/register.html', form=form)

    @app.route('/forgot')
    def forgot():
        form = ForgotForm(request.form)
        return render_template('forms/forgot.html', form=form)


    return app
