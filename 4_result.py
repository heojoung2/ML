# -*- coding: utf-8 -*-

from __future__ import unicode_literals     #unicode 문제 해결
import pycrfsuite                           #crf
import sys


def input_string_function():        #입력 스트링 받기
    print 'input : '

    input_string = raw_input()
    input_string = unicode(input_string)

    # 문자열 공백지우기
    result = ''

    for i in range(len(input_string)):
        if input_string[i] != ' ':
            result += input_string[i]

    return result


def feature_function(input_string):
    feature_string = []  # [[나,3:는],[는,2:나,3:먹],[먹,1:나,2:는,3:다],[는,1:는,2:먹,3:다],[다,1:먹,2:는]]

    for i in range(len(input_string)):
        syllable_feature = []  # [먹,1:나,2:는,3:다]
        syllable_feature.append(input_string[i])

        if i > 1:
            syllable_feature.append('1:' + input_string[i - 2])
        if i > 0:
            syllable_feature.append('2:' + input_string[i - 1])
        if i < len(input_string) - 1:
            syllable_feature.append('3:' + input_string[i + 1])

        feature_string.append(syllable_feature)

    return feature_string


def tag_function(features):     #태그 만들기
    crf_tagger = pycrfsuite.Tagger()    #입력 시퀀스의 레이블 시퀀스를 예측
    crf_tagger.open('crf.crfsuite')     # 모델 파일을 연다

    tag = crf_tagger.tag(features)        #항목 순서에 대한 레이블 순서를 예측
    crf_tagger.close()              #모델 닫기

    return tag


def combine_function(input_string, tag):        #입력 스트링과 태그 합쳐 결과내기
    result=''

    for i in range(len(input_string)):
        if tag[i]=='1':
            result+=(' '+input_string[i])
        else:
            result+=input_string[i]

    if result[0]==' ':      #결과의 처음이 띄어져있을 때
        return result[1:]
    else:
        return result

#최종 띄어쓰기 결과 함수
def spacing_function():
    input_string=input_string_function()
    features=feature_function(input_string)
    tag = tag_function(features)
    print combine_function(input_string, tag)
    return combine_function(input_string, tag)

#메인
reload(sys)
sys.setdefaultencoding('utf-8')

spacing_function()