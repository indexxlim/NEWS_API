import torch
import tqdm
import numpy as np
import os
import re
from konlpy.tag import Mecab


def correct_label(df):
    '''Label이 잘못된 부분을 수정하는 함수
    Args: 
        - df : Dataframe [내용, 평가]
    Return:
        - df : Label이 수정된 DataFrame'''
    
    correct_label_dict = {
        '긍정': '옹호',
        '긍정/보도자료': '옹호',
        '보도자료.옹호': '옹호',
        '중입': '중립',
        ' 중립': '중립',
        '옹호 ': '옹호',
        '증립': '중립',
        '주읿': '중립', 
        '중힙': '중립',
        '옹로': '옹호',
        '엉호': '옹호',
        '중랍': '중립',
        '융호': '옹호',
        '부부정': '부정',
        '옹ㅎ': '옹호',
        '부정 ': '부정',
        '웅호': '옹호',
        '올호': '옹호',
        '중립ㅂ': '중립',
        '부넝': '부정',
        '옹호옹호': '옹호',
        ' 부정': '부정',
        '부정정': '부정',
        '온호': '옹호',
        '즁립': '중립',
        '부덩': '부정',
        '부저ㅏㅇ': '부정',
        '옹ㅇ호': '옹호',
        '뷰종': '부정',
        '증ㄹ;ㅂ': '중립',
        'ㅈ우': '중립',
        '중리': '중립',
        '엉허': '옹호',
        '붱': '부정',
        '오옿': '옹호',
        '중': '중립',
        '즁랍': '중립',
        '즁': '중립',
        '증': '중립',
        '중ㅂ': '중립',
        '부': '부정',
        '부저': '부정',
        '부장': '부정', }

    df['평가'] = df['평가'].replace(correct_label_dict)
    # 옹호, 중립, 평가 데이터 만 들고오기
    df = df.query("(평가 == '옹호') | (평가 == '부정') | (평가 == '중립')")
    
    return df

def checkNAN(text):
    if isinstance(text, int):
        text = str(text)

    if text != text :
        text=''
        return text
    return text

def clean_text2(text):
    r = re.compile(r"""
        #e-mail 제거
        ([\w\d.]+@[\w\d.]+)|
        ([\w\d.]+@)|


        # 전화번호 제거#
        (\d{2,3})-(\d{3,4}-\d{4})|
        (\d{3,4}-\d{4})|

        (www.\w.+)|
        (.\w+.com)|
        (.\w+.co.kr)|
        (.\w+.co.kr)|
        (.\w+.go.kr)
        """,
    re.X | re.MULTILINE)
    r.sub(r'', text)
    return text

def clean_text(text):
    '''기사 내용 전처리 함수
    Args:
        - text: str 형태의 텍스트
    Return:
        - text: 전처리된 텍스트'''
    # Common
    # E-mail 제거#
    text = re.sub(r'([\w\d.]+@[\w\d.]+)', '', text)
    text = re.sub(r'([\w\d.]+@)', '', text)
    # 괄호 안 제거#
    text = re.sub(r"<[\w\s\d‘’=/·~:&,`]+>", "", text)
    text = re.sub(r"\([\w\s\d‘’=/·~:&,`]+\)", "", text)
    text = re.sub(r"\[[\w\s\d‘’=/·~:&,`]+\]", "", text)
    text = re.sub(r"【[\w\s\d‘’=/·~:&,`]+】", "", text)
    # 전화번호 제거#
    text = re.sub(r"(\d{2,3})-(\d{3,4}-\d{4})", "", text)  # 전화번호
    text = re.sub(r"(\d{3,4}-\d{4})", "", text)  # 전화번호
    # 홈페이지 주소 제거#
    text = re.sub(r'(www.\w.+)', '', text)
    text = re.sub(r'(.\w+.com)', '', text)
    text = re.sub(r'(.\w+.co.kr)', '', text)
    text = re.sub(r'(.\w+.go.kr)', '', text)
    text = re.sub(r'(.\w+.or.kr)', '', text)

    # 기자 이름 제거#
    text = re.sub(r"/\w+[=·\w@]+\w+\s[=·\w@]+", "", text)
    text = re.sub(r"\w{2,4}\s기자", "", text)
    # 한자 제거#
    text = re.sub(r'[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900]+', '', text)
    # 특수기호 제거#
    text = re.sub(r"[◇#/▶▲◆■●△①②③★○◎▽=▷☞◀ⓒ□?㈜♠☎]", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    # 따옴표 제거#
    text = re.sub(r"[\"\'”“‘’]", "", text)
    # 2안_숫자제거#
    # text = re.sub('[0-9]+',"",text)
    # 개행문자 제거#
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\\r\\n", "", text)


    return text

def morpheme(text):

    mecab = Mecab()
    text = [f'{word}_{pos}' for word, pos in mecab.pos(text)]

    return text


def label_encoder(df):
    '''Label Encodingto
    Args:
        - df: DataFrame
    Return:
        - df: DataFrame
            - 옹호 = 1, 부정 = -1, 중립 = 0'''
    df['평가'].replace('옹호', 1, inplace=True)
    df['평가'].replace('부정', -1, inplace=True)
    df['평가'].replace('중립', 0, inplace=True)
    return df
    
def make_path(file):
    curpath = os.getcwd()

    data_path = file.split('/')
    del data_path[-1]
    for i in data_path:    
        curpath = os.path.join(curpath, i)
        if not os.path.exists(curpath):
            os.mkdir(curpath)
    
def is_csv(file):
    if file.split('.')[-1] == 'csv':
        return True
    else:
        return False

# function
def softmax(z):
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()

def pad_sentences(sentences, sequence_length = 500, min_length=5):

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if (len(sentence) < min_length):
            new_sentence = np.zeros((sequence_length, 100), dtype=float)
        else:
            num_padding = sequence_length - len(sentence)
            if num_padding > 0 :
                padd = np.zeros((num_padding,100))
                new_sentence = np.vstack((sentence, padd))
            else :
                new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def pad_sentences2(sentences, sequence_length = 500):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding > 0 :
            padd = ['0'] * num_padding
            new_sentence = list(sentence) + padd
            new_sentence = np.array(new_sentence)
        else :
            new_sentence = sentence[:500]
        padded_sentences.append(new_sentence)
    return padded_sentences





class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)