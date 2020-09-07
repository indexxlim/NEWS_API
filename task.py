import torch
import time
from celery import Celery
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from konlpy.tag import Mecab

from app.main.newstone.CNN_Model import CNN
from app.main.newstone.utils  import softmax, Logger, correct_label, clean_text, label_encoder, make_path, is_csv, pad_sentences,checkNAN

from modelconfig import *


config_for_learning = dict(
	sentence_len=500,
	embedding_dim=100,
	num_filters = 100,
	kernel_sizes = [3,4,5],
	batch_size = 128,
	output_size = 3
)


app = Celery('task', backend='amqp', broker='amqp://guest@localhost:5672')
_model = CNN(500, config_for_learning['embedding_dim'], config_for_learning['num_filters'], kernel_size=config_for_learning['kernel_sizes'], stride=1,output_size=3)
_model.load_state_dict(torch.load(total_model))
_wv = Word2Vec.load(total_word2vec_path)

_gas_model = CNN(1000, config_for_learning['embedding_dim'], config_for_learning['num_filters'], kernel_size=config_for_learning['kernel_sizes'], stride=1,output_size=3)
_gas_model.load_state_dict(torch.load(gas_model))
_gas_wv = Word2Vec.load(new_word2vec_path)




print('just task synchronization')
mecab = Mecab()

app.conf.update(
    CELERY_TIMEZONE='Asia/Seoul',
    WORKER_MAX_MEMORY_PER_CHILD=20000000
)


@app.task
def sentiment(text, sequence_length):
    #task_spec = json.loads(json_str)
    #text = 
    #_wv = w2v.word2vec
    #_model = model


    start = time.time()  # 시작 시간 저장
    try:
        text = [checkNAN(i) for i in text]
     #  text = [clean_text(i) for i in text]
        text = list([f'{word}_{pos}' for word, pos in mecab.pos(text)] for text in text)

        final_data_setence = []
        for i in range(len(text)):
            tmp_sentence = []
            not_lists = []
            for j in range(len(text[i])):
                tmp = text[i][j]
                try:
                    tmp_sentence.append(_wv.wv[tmp])  # embedding[]
                except:
                    pass
            final_data_setence.append(np.array(tmp_sentence))

        final_data_setence = pad_sentences(final_data_setence, sequence_length = 500)

        x_test = np.array(final_data_setence, dtype=float)
        y_test = np.zeros(x_test.shape[0], dtype=float)
    except Exception as ex:
        return [], "preprocessing error! - "+ ex, -1




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

    except Exception as ex:
        return [], "Newral Network Test error! - "+ ex, -1

    neg = total_pred[:,0]
    nuet = total_pred[:,1]
    pos = total_pred[:,2]
    # performance
    '''성능 평가'''
    total_pred = np.argmax(total_pred, axis=1)
    #return total_pred
    total_pred = total_pred.tolist()
    
    
    return total_pred, 'OK' ,1, pos, nuet, neg



@app.task
def sentiment2(text):
    #task_spec = json.loads(json_str)
    #text =
    #_wv = w2v.word2vec
    #_model = model


    start = time.time()  # 시작 시간 저장
    try:
        text = [checkNAN(i) for i in text]
        #text = [clean_text(i) for i in text]
        text = list([f'{word}_{pos}' for word, pos in mecab.pos(text)] for text in text)

        final_data_setence = []
        for i in range(len(text)):
            tmp_sentence = []
            not_lists = []
            for j in range(len(text[i])):
                tmp = text[i][j]
                try:
                    tmp_sentence.append(_gas_wv.wv[tmp])  # embedding[]
                except:
                    pass
            final_data_setence.append(np.array(tmp_sentence))

        final_data_setence = pad_sentences(final_data_setence, sequence_length = 1000)

        x_test = np.array(final_data_setence, dtype=float)
        y_test = np.zeros(x_test.shape[0], dtype=float)
    except Exception as ex:
        return [], "preprocessing error! - "+ ex, -1



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
            preds, _, = _gas_model(inputs)
            preds = preds.cpu()
            tmp = softmax(preds)
            tmp = tmp.detach().numpy()
            total_pred = np.append(total_pred, tmp)
        total_pred = total_pred.reshape((-1, config_for_learning['output_size']))

    except Exception as ex:
        return [], "Newral Network Test error! - "+ ex, -1

    neg = total_pred[:, 0]
    nuet = total_pred[:, 1]
    pos = total_pred[:, 2]
    # performance
    '''성능 평가'''
    total_pred = np.argmax(total_pred, axis=1)
    # return total_pred
    total_pred = total_pred.tolist()

    return total_pred, 'OK' ,1, pos, nuet, neg
