# WIGO_API
Restful API for learning

# 개발환경
윈도우, anaconda, python version = 3.7


# Requirment
pytorch설치  
https://pytorch.org/ 에서 환경에 맞는 torch 설치  
__예__ 윈도우 anaconda 의 경우  `conda install pytorch torchvision cpuonly -c pytorch`

```pip install -r requirments.txt```

윈도우 환경이나 그 외에서 flask_monitoring 설치중 cp949 encoding 에러가 날 경우 git을 통해 직접 받은후
```bash
git clone https://github.com/flask-dashboard/Flask-MonitoringDashboard.git
cd Flask-MonitoringDashboard
python setup.py install
```
encoding='UTF8' 추가해주시면 됩니다.

__mecab__   
1. mecab-ko-msvc 설치하기 'C 기반으로 만들어진 mecab'이 윈도우에서 실행될 수 있도록 하는 역할  
1-1. 링크 클릭 https://github.com/Pusnow/mecab-ko-msvc/releases/tag/release-0.9.2-msvc-3  
1-2. 윈도우 버전에 따라 32bit / 64bit 선택하여 다운로드  
1-3. 'C 드라이브'에 mecab 폴더 만들기 => "C:/mecab"  
1-4. '1-2'에서 다운로드 받은 'mecab-ko-msvc-x64.zip' 또는 'mecab-ko-msvc-x84.zip' 압축풀기  
2. mecab-ko-dic-msvc.zip 기본 사전 설치하기  
2-1. 링크 클릭 https://github.com/Pusnow/mecab-ko-dic-msvc/releases/tag/mecab-ko-dic-2.1.1-20180720-msvc  
2-2. 사전 다운로드 'mecab-ko-dic-msvc.zip'  
2-3. 앞서 '1-3'에서 만들었던 "C:/mecab"에 압축해제  
* mecab 하위 폴더에 대강 파일들이 존재해야함  
3. python wheel 설치하기  
3-1. 링크 클릭 https://github.com/Pusnow/mecab-python-msvc/releases/tag/mecab_python0.996_ko_0.9.2_msvc-2  
3-2. 파이썬 및 윈도우 버전에 맞는 whl 다운로드  
나는 윈도우 64bit에 파이썬 3.7이여서 'mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl'다운로드  
3-3. 다운로드 받은 파일을 site-package 폴더에 옮겨놓기  
3-4. python 사용자의 경우 cmd창에서 site-package 폴더로 이동하여  
'pip install mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl'  
pip install 다운로드받은파일명.whl  
입력하여 설치 완료  
4. mecab 실행해보기  
4-1. 기본 소스 코드 넣어서 사용하기  
```bash
import MeCab
m = MeCab.Tagger()
out= m.parse("미캅이 잘 설치되었는지 확인중입니다.")
print(out)
```
4-2. 결과 확인하기  
미 NNP,인명,F,미,*,*,*,*  
캅 NNP,인명,T,캅,*,*,*,*  
이 JKS,*,F,이,*,*,*,*  
잘 MAG,*,T,잘,*,*,*,*  
설치 NNG,행위,F,설치,*,*,*,*  
되 XSV,*,F,되,*,*,*,*  
었 EP,*,T,었,*,*,*,*  
는지 EC,*,F,는지,*,*,*,*  
확인 NNG,행위,T,확인,*,*,*,*  
중 NNB,*,T,중,*,*,*,*  
입니다 VCP+EF,*,F,입니다,Inflect,VCP,EF,이/VCP/*+ᄇ니다/EF/*  
. SF,*,*,*,*,*,*,*  

출처 : https://cleancode-ws.tistory.com/97

mecab-ko-dic 경로를 못찾을경우 \site-packages\\konlpy\\tag\\_mecab.py 에서 
dicpath를 mecab_ko_dic경로로 바꾸면됩니다.

그 외의 환경에서는 bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh) 

# API 실행 및 request
API 실행은 python manage.py run
request 요청은 python request.py [최대 뉴스 갯수] [실행시킬 횟수] 


request는 현재 5만개의 데이터 중에서 몇개를 몇번 반복시킬지로 구성해놨습니다.
그에 대한 응답은 뉴스 4개일 경우 [2, 1, 1, 0] 이와 같이 각각 뉴스의 결과에 대해 나옵니다. 2는 긍정, 1은 중립, 0은 부정입니다.


# Celery를 통한 AMQP
celery -A task worker --pool=solo -l info

1. windows에서 Erlang과 Rabbitmq 설치
> rabbitmq-service.bat stop
> rabbitmq-service.bat install
> rabbitmq-service.bat start

task.py 파일에서 현재 감성분석에 대한 함수를 작성하였으며 이를 통해 celery로 프로세스들을 관리
미흡한 부분이 있을 수도 있습니다.



감사합니다.
