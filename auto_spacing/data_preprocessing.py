# -*- coding: utf-8 -*-
import csv

def tag_function(sentence):     #태그함수
    string=':'      #csv입력시 오류방지
    tag=''
    for i in range(len(sentence)):
        if i == 0:
            string += sentence[i]
            tag+='1 '
        elif sentence[i] != ' ' and sentence[i - 1] == ' ':  # 첫 음절일때
            string += sentence[i]
            tag+='1 '
        elif sentence[i] != ' ' and sentence[i - 1] != ' ':  # 아닐 때
            string += sentence[i]
            tag+='0 '

    return string,tag

def open_csv(num):  #전처리데이터 만들기
    csv_file = open('tt'+str(num)+'.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    return csv_writer,0

num=1
csv_writer, cnt = open_csv(num)
text_file = open('raw_text1.txt','r',encoding='utf-8')      #text1,text2,text3

for sentence in text_file:
    sentences = sentence.split('.')
    for i in sentences:
        i=i.strip()

        if len(i)>=10 and len(i)<=100:         #훈련 때 메모리방지
            if not '<' in i and not ';' in i:        #<>태그있는거 제외
                string,tag=tag_function(i)   #태그달기
                try:
                    csv_writer.writerow([string,tag])   #csv입력
                    cnt += 1
                    if cnt == 1000000:  # 백만기준으로 자르기
                        num += 1
                        csv_writer, cnt = open_csv(num)
                except:
                    pass

text_file.close()
