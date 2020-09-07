import os
import logging
import re
import pandas as pd
import datetime

file_list = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".log"):
            filepath = os.path.join(root, file)
            if filepath.split('\\')[-1].split('.')[0] == 'error':
                continue
            print(filepath)
            file_list.append(filepath)

log_format = '<Date> <Time>: <levelname> <Component>  <message>'



headers = []
splitters = re.split(r'(<[^<>]+>)', log_format)
regex = ''
for k in range(len(splitters)):
    if k % 2 == 0:
        splitter = re.sub(' +', '\\\s+', splitters[k])
        regex += splitter
    else:
        header = splitters[k].strip('<').strip('>')
        regex += '(?P<%s>.*?)' % header
        headers.append(header)
regex = re.compile('^' + regex + '$')

all_logdf =  pd.DataFrame(columns=['Date', 'Time', 'levelname', 'Component', 'message', 'processname'])

for i in file_list:
    log_messages = []
    linecount = 0
    with open(i, 'r',encoding='utf-8') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip().replace('[', '', 1).replace(']', '', 1).replace('/', ' ', 1))
                message = [match.group(header) for header in headers]
                message.append(i.split('\\')[-1])
                datetime.datetime.strptime(message[0], "%Y-%m-%d").strftime("%Y-%m-%d")

                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=['Date', 'Time', 'levelname', 'Component', 'message', 'processname'])
    all_logdf = all_logdf.append(logdf)
    
    
all_logdf = all_logdf.sort_values('Date')

all_logdf.to_csv('logdf.csv', encoding='utf-8-sig', index=False)
