import pickle
import numpy as np
import pandas as pd
from app.main.newstone.sentdic import sentdic

from ..newstone.utils import softmax, Logger, correct_label, clean_text, label_encoder, make_path, is_csv, pad_sentences
from konlpy.tag import Mecab

from sklearn.metrics import confusion_matrix, f1_score


def assign_unique_freq(words, article):
    word2pos = {w: s for w, s in zip(words['word'], words['score']) if s > 0}
    word2neg = {w: s for w, s in zip(words['word'], words['score']) if s < 0}
    score = [sum([1 if x in word2pos else -1 if x in word2neg else 0 for x in set(a)]) for a in article]
    score_assigned = [{x:word2pos[x] if x in word2pos else word2neg[x] if x in word2neg else 0 for x in set(a)} for a in article]
    return score_assigned, [1 if a > 0 else -1 if a < 0 else 0 for a in score]


def sent_analysis(text):
    _sentdic = sentdic.sentdic[1]

    try:
        text = [clean_text(i) for i in text]
        mecab = Mecab()

        text = list([word for word, pos in mecab.pos(text)] for text in text)
        text = np.array(text)
    except:
        return 'preprocess error!', -1

    try:
        wordsenti_score, article_sent = assign_unique_freq(_sentdic, text)
    except:
        return 'wordsenti error!', -1


    return wordsenti_score, 1


def sent_analysis_second(text):
    _sentdic = sentdic.sentdic[1]

    text = [clean_text(i) for i in text]
    mecab = Mecab()

    text = list([word for word, pos in mecab.pos(text)] for text in text)
    text = np.array(text)


    wordsenti_score, article_sent = assign_unique_freq(_sentdic, text)


    return wordsenti_score