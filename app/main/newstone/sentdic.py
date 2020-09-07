from flask import current_app

from gensim.models import Word2Vec

import gensim
import numpy as np
import pandas as pd

# Find the stack on which we want to store the database connection.
# Starting with Flask 0.9, the _app_ctx_stack is the correct one,
# before that we need to use the _request_ctx_stack.
try:
    from flask import _app_ctx_stack as stack
except ImportError:
    from flask import _request_ctx_stack as stack
    
 
def load_word(word_path, n=None):
    word = pd.read_pickle(word_path)
    #word = pd.read_csv(word_path, index_col=0, encoding="ANSI")
    word.columns = ['word', 'pos', 'neg', 'used_words']
    # null 인 부분을 0으로
    word.loc[pd.isnull(word['pos']), 'pos'] = 0
    word.loc[pd.isnull(word['neg']), 'neg'] = 0
    word['score'] = word['pos'] - word['neg']

    # 긍 부정 각각 상위 n개 뽑을 때
    if n is not None:
        sort_words = word.sort_values(by='score', ascending=True).reset_index(drop=True)
        word = pd.concat([sort_words[:n], sort_words[-n:]], ignore_index=True)

    word['word'] = list(map(lambda x: x.split("_")[0], word['word']))

    return word


class FlaskSentDic(object):
    
    _delidx = {}
    _labelspread = {}

    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        app.config.setdefault('DELIDX_FILE', 'delidx.npy')
        app.config.setdefault('LABELSPREAD_FILE', 'labelspreading_result.pkl')



        delidx_path = app.config['delidx_path']
        labelspread_path = app.config['labelspread_path']

    
            
        self._delidx[app.config['DELIDX_FILE']] = np.load(delidx_path)
        self._labelspread[app.config['LABELSPREAD_FILE']] = load_word(labelspread_path)
        print('load labelspread and delidx(for sentdic)')

    

    @property
    def sentdic(self):
        ctx = stack.top
        if ctx is not None:
            if not hasattr(ctx, '_delidx'):
                ctx._delidx = self._delidx[current_app.config['DELIDX_FILE']]
            if not hasattr(ctx, '_labelspread'):
                ctx._labelspread = self._labelspread[current_app.config['LABELSPREAD_FILE']]
            return ctx._delidx, ctx._labelspread
        
sentdic = FlaskSentDic()