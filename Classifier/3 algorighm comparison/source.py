# -*- coding: utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt

def string_to_float(list):  # 스트링을 플루트형으로 변환
    result = []
    for i in range(len(list)):
        result.append(float(list[i]))
    return result

def Y_matrix(first_class,second_class):         #Y행렬로 합치는 함수
    class1 = copy.copy(first_class)
    class1 = np.array(class1)

    class2 = copy.copy(second_class)
    class2 = -np.array(class2)

    one_array = np.array([([1 for i in range(10)] + [-1 for i in range(10)])])  # 1-10개, -1-10개인 일차원 배열
    temp_class = np.concatenate((class1, class2), axis=0)
    result = np.concatenate((one_array.T, temp_class), axis=1)
    return result

def perceptron(weight, first_class, second_class):      #perceptron
    array_weight=np.array([weight])                                     #a행렬
    train_sample=Y_matrix(first_class,second_class)                     #Y행렬
    discriminant = np.dot(array_weight,train_sample.T)[0]               #g(x)

    bool_array = (discriminant<=0)                      #criterion 판별
    criterion_array = train_sample[bool_array,:]

    criterion=[0,0,0]
    for i in criterion_array:
        for j in range(3):
            criterion[j]+=i[j]

    return criterion

def relaxtion(weight, first_class, second_class,margin_b):          #relaxtion
    array_weight=np.array([weight])                                     #a행렬
    train_sample=Y_matrix(first_class,second_class)                     #Y행렬
    discriminant = np.dot(array_weight,train_sample.T)[0]               #g(x)

    bool_array = (discriminant<=0)                              #criterion 판별
    criterion_array = train_sample[bool_array,:]

    criterion=[0,0,0]
    for i in criterion_array:
        temp=np.array([i])         #(1,3) 행렬을 만들어주기위해
        result= ((margin_b-np.dot(array_weight,temp.T))[0][0]/(pow(i[0],2)+pow(i[1],2)+pow(i[2],2)))*i

        for j in range(3):
            criterion[j]+=result[j]

    return criterion

def widrow_hoff(weight,first_class,second_class,margin_vector):          #Widrow-hoff(LMS)
    array_weight=np.array([weight])                                     #a행렬
    train_sample=Y_matrix(first_class,second_class)                     #Y행렬
    discriminant = np.dot(array_weight,train_sample.T)[0]               #g(x)
    margin_vector = np.array(margin_vector)                     #margin_vector 행렬

    bool_array = (discriminant<=0)                      #criterion 판별
    criterion_array = train_sample[bool_array,:]
    margin_vector=margin_vector[bool_array,:]

    criterion = np.dot((margin_vector- np.dot(array_weight,criterion_array.T)),criterion_array)
    return criterion[0]

def plot_graph(first_class,second_class,weight,number):         #도식화
    plt.figure()
    title='number '+str(number)
    plt.title(title)
    for i in range(len(class1)):
        plt.plot(first_class[i][0], first_class[i][1], 'ro')
        plt.plot(second_class[i][0], second_class[i][1], 'bo')

    xlist, ylist = [np.linspace(-7.0, 10.0), np.linspace(-7.0, 10.0)]
    X, Y = np.meshgrid(xlist, ylist)
    Z = weight[1] * X + weight[2] * Y + weight[0]
    plt.contour(X, Y, Z, 0)


class1, class2, class3 = [[], [], []]  # 클래스들

infile = open('train.txt', 'r')  # 파일읽기
for i in infile:  # 클래스별 훈련데이터 분배
    i = i.strip('\n').split(' ')
    if i[0] == '1':
        class1.append(string_to_float(i[1:]))
    elif i[0] == '2':
        class2.append(string_to_float(i[1:]))
    elif i[0] == '3':
        class3.append(string_to_float(i[1:]))

# 1번
weight = [0.5, 0.5, -0.5]  # bias, 가중치
learning = 0.01  # 학습률
threshold = 0  # 임계값
print '1번'
print 'first_w0:' + str(weight[0])+', first_w1:' + str(weight[1]) + ', first_w2:' + str(weight[2]) + ', learnging:' + str(learning) + ', threshold:' + str(threshold)

while True:
    criterion = perceptron(weight, class1, class2)      #기준식

    for i in range(3):
        weight[i]= weight[i] + learning * criterion[i]

    if abs(learning * criterion[0]) <= threshold and abs(learning * criterion[1]) <= threshold and abs(learning * criterion[2]) <= threshold:
        break

plot_graph(class1,class2,weight,1)
print 'first_w0:' + str(weight[0])+', final_w1:' + str(weight[1]) + ', final_w2:' + str(weight[2]) + '\n'
'''

# 2번
weight = [0, 0, 0]  # bias, 가중치
margin_b = 0.5      #margin_b
learning = 0.01  # 학습률
threshold = 2000  # 임계값
print '2번'
print 'first_w0:' + str(weight[0])+', first_w1:' + str(weight[1]) + ', first_w2:' + str(weight[2]) + ', margin_b:' + str(margin_b)+ ', learnging:' + str(learning)+', threshold:' + str(threshold)

cnt=0
while True:
    criterion = relaxtion(weight, class1, class3,margin_b)      # 기준식

    for i in range(3):
        weight[i]= weight[i] + learning * criterion[i]

    if criterion[0]==0 and criterion[1]==0 and criterion[2]==0:
        break
    elif cnt==threshold:
        print '-endless loop-'
        break
    cnt+=1

plot_graph(class1,class3,weight,2)
print 'final_w0:' + str(weight[0])+', final_w1:' + str(weight[1]) + ', final_w2:' + str(weight[2]) + '\n'


#3번
weight = [0.5, -0.5, -0.5]  # bias, 가중치
margin_vector = [[0.1]for i in range(20)]      #margin_b
threshold = 0.1       # 임계값
learning = 0.01     # 학습률

print '3번'
print 'first_w0:' + str(weight[0]) + ', first_w1:' + str(weight[1]) + ', first_w2:' + str(weight[2])+ ', margin_vector:' + str(margin_vector[0])+ ', threshold:' + str(threshold) + ', learnging:' + str(learning)

while True:
    criterion = widrow_hoff(weight,class1,class3,margin_vector)  # 기준식

    for i in range(3):
        weight[i] = weight[i] + learning * criterion[i]

    if learning*criterion[0]<=threshold and learning*criterion[1]<=threshold and learning*criterion[2]<=threshold:
        break

plot_graph(class1,class3,weight,3)
print 'final_w0:' + str(weight[0]) + ', final_w1:' + str(weight[1]) + ', final_w2:' + str(weight[2])

#plt.show()
'''
