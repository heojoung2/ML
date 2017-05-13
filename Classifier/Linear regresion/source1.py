#-*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import *

def string_list_float(features):      #스트링을 플루트으로 바꿔 리스트 만들기
    result=[]
    for i in features:
        result.append(float(i))
    return result

def mean(Iris) :    #array 평균 구하기
    result = np.array(Iris)
    return np.mean(result,axis=0)   #열 평균

def covariance(Iris,mean) :      #array 공분산 구하기
    result = np.array(Iris)
    C=[result[0:,i:i+1] for i in range(4)]
    result2=[[np.dot(C[i].T,C[j])[0][0]/len(Iris)-mean[i]*mean[j] for j in range(4)] for i in range(4)]
    return np.array(result2)

def discriminant(x,Iris_mean,Iris_covariance,i):      #판별식
    x=np.array(x)
    result1 = np.dot(np.dot(x.T,-1/2.0*inv(Iris_covariance[i])),x)
    result2 = np.dot(np.dot(inv(Iris_covariance[i]),Iris_mean[i]).T,x)
    result3 = -1/2.0*np.dot(np.dot(Iris_mean[i].T,inv(Iris_covariance[i])),Iris_mean[i])-1/2.0*np.log(np.linalg.det(Iris_covariance[i]))
    return result1+result2+result3

def boundary(x,Iris_mean,Iris_covariance,i,j):     #경계식
    x=np.array(x)
    result1 = np.dot(np.dot(x.T, -1 / 2.0 * (inv(Iris_covariance[i])-inv(Iris_covariance[j]))), x)
    result2 = np.dot((np.dot(inv(Iris_covariance[i]), Iris_mean[i]).T-np.dot(inv(Iris_covariance[j]), Iris_mean[j]).T), x)
    result3 = -1 / 2.0 * np.dot(np.dot(Iris_mean[i].T, inv(Iris_covariance[i])), Iris_mean[i]) - 1 / 2.0 * np.log(np.linalg.det(Iris_covariance[i]))
    result4 = -1 / 2.0 * np.dot(np.dot(Iris_mean[j].T, inv(Iris_covariance[j])), Iris_mean[j]) - 1 / 2.0 * np.log(np.linalg.det(Iris_covariance[j]))
    return result1+result2+(result3-result4)

Setosa, Versicolor, Virginica=[[],[],[]]
W=[Setosa,Versicolor,Virginica]

#훈련파일읽기
infile=open("Iris_train.dat")
for i in infile:
    try:
        features = i.split()
        if features[-1]=='1':
            W[0].append(string_list_float(features[:4]))
        elif features[-1]=='2':
            W[1].append(string_list_float(features[:4]))
        elif features[-1]=='3':
            W[2].append(string_list_float(features[:4]))
    except:
        pass
infile.close()

#1) a
Iris_mean = [mean(W[0]),mean(W[1]),mean(W[2])]
Iris_covariance = [covariance(W[0],Iris_mean[0]),covariance(W[1],Iris_mean[1]),covariance(W[2],Iris_mean[2])]

#1)b, c
result=[[0 for j in range(3)] for i in range(3)]
infile=open("Iris_test.dat")
for i in infile:
    try:
        features = i.split()
        features_float =string_list_float(features[:4])
        #print boundary(features_float, Iris_mean, Iris_covariance, 0, 1)  # b. 경계선값비교
        c = [discriminant(features_float, Iris_mean, Iris_covariance, j) for j in range(3)]
        if features[-1]=='1':
            result[0][c.index(max(c))]+=1
        elif features[-1]=='2':
            result[1][c.index(max(c))]+=1
        elif features[-1]=='3':
            result[2][c.index(max(c))]+=1
    except:
        pass
infile.close()
print np.array(result)  #c.

