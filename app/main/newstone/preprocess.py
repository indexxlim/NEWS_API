import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from konlpy.tag import Mecab
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
# ################local file##################
from app.main.newstone.config import config_for_learning
from app.main.newstone.utils  import softmax, Logger, correct_label, clean_text, label_encoder, make_path, is_csv, pad_sentences,checkNAN
from app.main.newstone.CNN_Model import dmodel
from app.main.newstone.word2vec import w2v
import time

def predict(text, output='NewsTone/result.txt'):
    # random seed / cuda setting

    _wv = w2v.word2vec
    _model = dmodel.CNN_Model
    mecab = Mecab()


    start = time.time()  # 시작 시간 저장
    try:
        text = [checkNAN(i) for i in text]
        text = [clean_text(i) for i in text]
        text = list([f'{word}_{pos}' for word, pos in mecab.pos(text)] for text in text)

        final_data_setence = []
        for i in range(len(text)):
            tmp_sentence = []
            not_lists = []
            for j in range(len(text[i])):
                tmp = text[i][j]
                try:
                    tmp_sentence.append(_wv[tmp])  # embedding[]
                except:
                    pass
            final_data_setence.append(np.array(tmp_sentence))

        final_data_setence  = pad_sentences(final_data_setence)

        x_test = np.array(final_data_setence, dtype=float)
        y_test = np.zeros(x_test.shape[0], dtype=float)
    except:
        return [], "preprocessing error!", -1


    del final_data_setence

    print("time this requests took :", time.time() - start)


    #print('--------------Data Loading Done------------')

    use_cuda = torch.cuda.is_available()
    #print('use_cuda = {}\n'.format(use_cuda))

    # test
    try:
        dataset_test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        test_loader = DataLoader(dataset_test, batch_size=config_for_learning['batch_size'], shuffle=False, num_workers=0,
                                 pin_memory=False)
        total_pred = np.array([])


        for i, (inputs, _) in enumerate(test_loader):
            if use_cuda:
                inputs = inputs.float().cuda()
            else:
                inputs = inputs.float()  # .cuda()
            preds, _, = _model(inputs)
            preds = preds.cpu()
            tmp = softmax(preds)
            tmp = tmp.detach().numpy()
            total_pred = np.append(total_pred, tmp)
        total_pred = total_pred.reshape((-1, config_for_learning['output_size']))

    except:
        return [], "Newral Network Test error!", -1

    # performance
    '''성능 평가'''
    total_pred = np.argmax(total_pred, axis=1)
    return total_pred, 'OK' ,1


def vec_predict(x_test, output='NewsTone/result.txt'):
    # random seed / cuda setting
    _wv = w2v.word2vec
    _model = dmodel.CNN_Model

    x_test = np.array(x_test, dtype=float)
    y_test = np.zeros(x_test.shape[0], dtype=float)

    print('--------------Data Loading Done------------')

    use_cuda = torch.cuda.is_available()
    print('use_cuda = {}\n'.format(use_cuda))

    # test
    dataset_test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = DataLoader(dataset_test, batch_size=config_for_learning['batch_size'], shuffle=False, num_workers=0,
                             pin_memory=False)
    total_pred = np.array([])

    for i, (inputs, _) in enumerate(test_loader):
        if use_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()  # .cuda()
        preds, _, = _model(inputs)
        preds = preds.cpu()
        tmp = softmax(preds)
        tmp = tmp.detach().numpy()
        total_pred = np.append(total_pred, tmp)
    total_pred = total_pred.reshape((-1, config_for_learning['output_size']))

    # performance
    '''성능 평가'''
    total_pred = np.argmax(total_pred, axis=1)
    return total_pred
