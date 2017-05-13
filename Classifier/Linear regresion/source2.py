#-*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

def string_list_float(features):      #스트링을 플루트로 바꿔 리스트 만들기
    result=[]
    for i in features:
        result.append(float(i))
    return result

def plot_train_data(W,color):       #훈련 데이터 그래프 띄우기
    x,y = [[],[]]
    for i in W:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x, y, color)

def mean(Iris) :    #array 평균 구하기
    result = np.array(Iris)
    return np.mean(result,axis=0)   #열 평균

def covariance(Iris,mean) :      #array 공분산 구하기
    result = np.array(Iris)
    C=[result[0:,i:i+1] for i in range(2)]
    result2=[[np.dot(C[i].T,C[j])[0][0]/len(Iris)-mean[i]*mean[j] for j in range(2)]for i in range(2)]
    return np.array(result2)

def mahalanobis(Iris_mean,Iris_covariance,i,color):      #마할라노비스거리
    xlist, ylist = [np.linspace(4.0, 8.0), np.linspace(1.5, 4.5)]
    X, Y = np.meshgrid(xlist, ylist)
    a,b,c,d =[inv(Iris_covariance[i])[0][0],inv(Iris_covariance[i])[0][1],inv(Iris_covariance[i])[1][0],inv(Iris_covariance[i])[1][1]]
    Z =(X-Iris_mean[i][0])*(a*(X-Iris_mean[i][0])+c*(Y-Iris_mean[i][1]))+(Y-Iris_mean[i][1])*(b*(X-Iris_mean[i][0])+d*(Y-Iris_mean[i][1]))-2
    plt.contour(X, Y, Z, 0, colors=color)

def boundary(Iris_mean,Iris_covariance,i,j,color):     #경계식
    xlist, ylist = [np.linspace(4.0, 8.0), np.linspace(1.5, 4.5)]
    X, Y = np.meshgrid(xlist, ylist)

    A = inv(Iris_covariance[i])[0][0] - inv(Iris_covariance[j])[0][0]
    B = inv(Iris_covariance[i])[0][1] - inv(Iris_covariance[j])[0][1]
    C = inv(Iris_covariance[i])[1][0] - inv(Iris_covariance[j])[1][0]
    D = inv(Iris_covariance[i])[1][1] - inv(Iris_covariance[j])[1][1]
    Z1 = -1/2.0*(X*(X*A+Y*C)+Y*(X*B+Y*D))

    A = inv(Iris_covariance[i])[0][0]*Iris_mean[i][0]+inv(Iris_covariance[i])[0][1]*Iris_mean[i][1]
    B = inv(Iris_covariance[j])[0][0]*Iris_mean[j][0]+inv(Iris_covariance[j])[0][1]*Iris_mean[j][1]
    C = inv(Iris_covariance[i])[1][0]*Iris_mean[i][0]+inv(Iris_covariance[i])[1][1]*Iris_mean[i][1]
    D = inv(Iris_covariance[j])[1][0]*Iris_mean[j][0]+inv(Iris_covariance[j])[1][1]*Iris_mean[j][1]
    Z2 = X*(A-B)+Y*(C-D)

    Z3 = -1 / 2.0 * np.dot(np.dot(Iris_mean[i].T, inv(Iris_covariance[i])), Iris_mean[i]) - 1 / 2.0 * np.log(np.linalg.det(Iris_covariance[i]))
    Z4 = -1 / 2.0 * np.dot(np.dot(Iris_mean[j].T, inv(Iris_covariance[j])), Iris_mean[j]) - 1 / 2.0 * np.log(np.linalg.det(Iris_covariance[j]))

    plt.contour(X, Y, Z1+Z2+(Z3-Z4), 0,colors=color)

def boundary_compare(x,Iris_mean,Iris_covariance,i,j):     #경계식비교
    x=np.array(x)
    result1 = np.dot(np.dot(x.T, -1 / 2.0 * (inv(Iris_covariance[i])-inv(Iris_covariance[j]))), x)
    result2 = np.dot((np.dot(inv(Iris_covariance[i]), Iris_mean[i]).T-np.dot(inv(Iris_covariance[j]), Iris_mean[j]).T), x)
    result3 = -1 / 2.0 * np.dot(np.dot(Iris_mean[i].T, inv(Iris_covariance[i])), Iris_mean[i]) - 1 / 2.0 * np.log(np.linalg.det(Iris_covariance[i]))
    result4 = -1 / 2.0 * np.dot(np.dot(Iris_mean[j].T, inv(Iris_covariance[j])), Iris_mean[j]) - 1 / 2.0 * np.log(np.linalg.det(Iris_covariance[j]))
    return result1+result2+(result3-result4)

def decision_plot(features,i):      #제대로 분류되었는지 판단
    if features[4] == i:
        plt.plot(features[0], features[1], color[i-1] + 'o')
    else:
        plt.plot(features[0], features[1], 'yo')

def test_plot(features,Iris_mean,Iris_covariance,color):      #테스트 데이터 띄우기
    if boundary_compare([features[0], features[1]],Iris_mean,Iris_covariance,0,1) >0:
        if boundary_compare([features[0], features[1]],Iris_mean,Iris_covariance,0,2) >0:
            decision_plot(features,1)
        else:
            decision_plot(features, 3)
    else:
        if boundary_compare([features[0], features[1]],Iris_mean,Iris_covariance,1,2) >0:
            decision_plot(features, 2)
        else:
            decision_plot(features, 3)

def discriminant(x,Iris_mean,Iris_covariance,i):      #판별식
    x = np.array(x)
    result1 = np.dot(np.dot(x.T, -1 / 2.0 * inv(Iris_covariance[i])), x)
    result2 = np.dot(np.dot(inv(Iris_covariance[i]), Iris_mean[i]).T, x)
    result3 = -1 / 2.0 * np.dot(np.dot(Iris_mean[i].T, inv(Iris_covariance[i])), Iris_mean[i]) - 1 / 2.0 * np.log(np.linalg.det(Iris_covariance[i]))
    return result1 + result2 + result3

Setosa, Versicolor, Virginica =[[],[],[]]
W=[Setosa,Versicolor,Virginica]
color=['r','b','g']

#훈련파일읽기
infile=open("Iris_train.dat")
for i in infile:
    try:
        features = i.split()
        if features[-1]=='1':
            W[0].append(string_list_float(features[:2]))
        elif features[-1]=='2':
            W[1].append(string_list_float(features[:2]))
        elif features[-1]=='3':
            W[2].append(string_list_float(features[:2]))
    except:
        pass
infile.close()

#a.

cnt=0
for i in W:
    plot_train_data(i,color[cnt]+'s')
    cnt+=1

#b.
Iris_mean = [mean(W[0]),mean(W[1]),mean(W[2])]
Iris_covariance = [covariance(W[0],Iris_mean[0]),covariance(W[1],Iris_mean[1]),covariance(W[2],Iris_mean[2])]

#c.
cnt=0
for i in range(3):
    mahalanobis(Iris_mean, Iris_covariance, i,color[cnt])
    plt.plot(Iris_mean[i][0],Iris_mean[i][1],'md')
    cnt+=1

#d.
cnt=0
for i in range(3):
    for j in range(3):
        if i<j:
            boundary(Iris_mean, Iris_covariance, i, j,color[cnt])
            cnt+=1

#e. f.
result=[[0 for j in range(3)] for i in range(3)]
infile=open("Iris_test.dat")
for i in infile:
    try:
        features = i.split()
        features_float =string_list_float(features[:2])
        test_plot(string_list_float(features),Iris_mean,Iris_covariance,color)  #e.
        res = [discriminant(features_float, Iris_mean, Iris_covariance, j) for j in range(3)]   #f.
        if features[-1]=='1':
            result[0][res.index(max(res))]+=1
        elif features[-1]=='2':
            result[1][res.index(max(res))]+=1
        elif features[-1]=='3':
            result[2][res.index(max(res))]+=1
    except:
        pass
infile.close()
print np.array(result)  #f.

plt.show()
