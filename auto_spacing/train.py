# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
import os

def padding(x_batch,y_batch,length):      #패딩
    max_length=max(length)
    for i in length:    #length=[1,2,3,4,5]
        for j in x_batch:   #X 패딩하기
            for k in range(len(j),max_length):
                j.append(0)
        for j in y_batch:   #Y 패딩하기
            for k in range(len(j), max_length):
                j.append(0)

    return x_batch,y_batch

def batch_function(cnt):  # 배치함수
    x_result = []   #한글
    x_batch = []  # 입력 [[[1,0,0],[1,0,0]]] -> shape(1,2,3) -> batch_size, sequence_length, input_dimension
    y_batch = []  # 출력 [[[x,x],[x,x]]] -> shape(1,2,2) -> 2 : batch_size, sequence_length, hidden_size
    lengths=[]

    for i in range(cnt, cnt + batch_size):
        string = csv_list[i][0][1:]
        x_result.append(string)
        lengths.append(len(string))
        x_data = []

        for j in string:  # 입력문장, 처음: 제거
            try:
                x_data.append(syllabe_dic[j])
            except:  # 사전에 없을 때
                x_data.append(0)

        x_batch.append(x_data)

        y_data = csv_list[i][1].split(' ')[:-1]  # 태그, 마지막'' 제거
        y_data = list(map(int, y_data))
        y_batch.append(y_data)

    x_batch,y_batch = padding(x_batch,y_batch,lengths)
    return x_result, x_batch, y_batch, lengths, max(lengths)

def spacing_result_function(x_result,y_result):     #최종결과
    string_list=[]
    for i in range(len(x_result)):
        string = ''
        for j in range(len(x_result[i])):
            if y_result[i][j]==1:
                string+=(' '+x_result[i][j])
            elif y_result[i][j]==0:
                string+=x_result[i][j]
        string_list.append(string.strip())
    print (string_list)

def open_csv(num):
    train_csv = open('tt'+str(num)+'.csv', 'r')
    csv_reader = csv.reader(train_csv)
    return list(csv_reader)

#variable
dir = os.path.dirname(os.path.realpath(__file__))

syllabe_csv = open('syllabe.csv','r')
csv_reader = csv.reader(syllabe_csv)
syllabe_list = list(csv_reader)
syllabe_list=np.squeeze(syllabe_list)

syllabe_dic = {n: i for i, n in enumerate(syllabe_list)}
syllabe_dic_len = len(syllabe_dic)  # 사전 크기
syllabe_csv.close()

hidden_size = 2
layers = 2
input_dim = syllabe_dic_len  # one-hot size

# 입력값
X = tf.placeholder(tf.int32, [None, None])  # X one-hot, [batch_size,seqeunce_length]
Y = tf.placeholder(tf.int32, [None, None])  # Y label, [batch_size, seqeuence_length]
Lengths = tf.placeholder(tf.int32, [None])  #length array, [batch_size]
Max_length = tf.placeholder(tf.int32)
Batch = tf.placeholder(tf.int32)
Keep_prob = tf.placeholder(tf.float32)

#one-hot encoding
X_one_hot = tf.one_hot(X,input_dim)

# RNN 구축
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)  # num_units=출력사이즈
cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=Keep_prob)
cell = tf.contrib.rnn.MultiRNNCell([cell]*layers, state_is_tuple=True)
initial_state = cell.zero_state(Batch, tf.float32)  # 초기 스테이트
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot,sequence_length=Lengths, initial_state=initial_state, dtype=tf.float32)

#softmax
X_for_softmax = tf.reshape(outputs,[-1,hidden_size])    #펼쳐진것을 하나로 합친다.
softmax_w = tf.get_variable("softmax_w",[hidden_size,hidden_size],initializer = tf.contrib.layers.xavier_initializer())
softmax_b = tf.get_variable("softmax_b",[hidden_size],initializer = tf.contrib.layers.xavier_initializer())
outputs = tf.matmul(X_for_softmax,softmax_w)+softmax_b      #softmax outputs
outputs = tf.reshape(outputs,[Batch,Max_length,hidden_size])  #하나로 합친것을 다시 펼친다

# Cost
weights = tf.ones([Batch, Max_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs,axis=2)

#save graph
saver = tf.train.Saver()

# session 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())                 #처음
#saver.restore(sess, tf.train.latest_checkpoint('./'))      #이어서
#saver.restore(sess,'./ckpt/my-model-3' )

# training
batch_size=50
for epoch in range(5):  #epoch
    for i in range(0,5):      #traing dataset 13
        csv_list = open_csv(i+1)

        for j in range(0,len(csv_list),batch_size):     #batch
            x_result, x_batch, y_batch, lengths, max_length = batch_function(j)

            try:
                _,Loss = sess.run([train,loss], feed_dict={X: x_batch, Y: y_batch, Lengths: lengths, Max_length:max_length, Batch:batch_size, Keep_prob : 0.9})
                y_result = sess.run(prediction, feed_dict={X: x_batch, Lengths:lengths, Max_length:max_length, Batch:batch_size, Keep_prob: 1.0})
                spacing_result_function(x_result, y_result)
                print(epoch, ' ', i,' ', j,' ',Loss)
            except:
                print('error')
    saver.save(sess, dir + '/ckpt/my-model',global_step=(epoch))
