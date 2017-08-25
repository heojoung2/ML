#-*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
import math

def read_train_data():
    with h5py.File('train_data.hf', 'r') as hf:
        X = np.array(hf["X"])
        Y_colorization = np.array(hf["Y_colorization"])
        Y_classification = np.array(hf["Y_classification"])


    return X, Y_colorization, Y_classification

def Convolution(input, weight_name, bias_name, input_num, output_num, stride):
    weight = tf.get_variable(weight_name, shape=[3,3,input_num,output_num],initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(bias_name, shape=[output_num], initializer=tf.contrib.layers.xavier_initializer())
    convolution = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding="SAME") + bias

    if weight_name=="colorization_network1_w7":
        sigmoid = tf.nn.sigmoid(convolution)
        return sigmoid

    relu = tf.nn.relu(convolution)
    return relu

def Fully_connected(input, weight_name, bias_name, input_num, output_num):
    weight = tf.get_variable(weight_name,shape=[input_num,output_num],initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(bias_name,shape=[output_num],initializer=tf.contrib.layers.xavier_initializer())

    if weight_name=="global_level_w5":
        input = tf.reshape(input,[-1,input_num])

    matmul_matrix = tf.matmul(input,weight)+bias
    relu = tf.nn.relu(matmul_matrix)
    return relu

def Fusion(input1,input2,weight_name,bias_name):
    input2 = tf.tile(input2,[1,28*28])
    input2 = tf.reshape(input2,[2,28,28,256])
    input1 = tf.concat([input1,input2],3)
    weight = tf.get_variable(weight_name, shape=[1,1,512,256], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(bias_name, shape=[256], initializer=tf.contrib.layers.xavier_initializer())
    convolution = tf.nn.conv2d(input1, weight, strides=[1, 1, 1, 1], padding="SAME")+bias
    relu = tf.nn.relu(convolution)
    return relu

def Upsample(input,weight_name,height_size,width_size, channel):
    weight = tf.get_variable(weight_name, shape=[2, 2, channel, channel], initializer=tf.contrib.layers.xavier_initializer())
    upconvolution = tf.nn.conv2d_transpose(input,weight,[batch_size, height_size, width_size, channel],[1,2,2,1],padding="SAME",name=None)
    return upconvolution

class Model:
    def __init__(self, sess):
        self.sess = sess
        self.build_network()

    def build_network(self):
        image_height_size = 224
        image_width_size = 224
        classification_num = 2 #205
        loss_parameter = 1/300
        learning_rate = 0.001

        self.X = tf.placeholder(tf.float32, [batch_size,image_height_size,image_width_size])
        self.Y_colorization = tf.placeholder(tf.float32, [batch_size,image_height_size,image_width_size,2])
        self.Y_classification = tf.placeholder(tf.int32, [batch_size])
        X = tf.reshape(self.X, [batch_size,image_height_size,image_width_size,1])
        Y_classification_one_hot = tf.one_hot(self.Y_classification, classification_num)

        low_level1 = Convolution(input=X, weight_name = "low_level_w1", bias_name = "low_level_b1", input_num=1, output_num = 64, stride=2)
        low_level2 = Convolution(input=low_level1, weight_name="low_level_w2", bias_name = "low_level_b2", input_num=64, output_num=128, stride=1)
        low_level3 = Convolution(input=low_level2, weight_name="low_level_w3", bias_name = "low_level_b3", input_num=128, output_num=128, stride=2)
        low_level4 = Convolution(input=low_level3, weight_name="low_level_w4", bias_name = "low_level_b4", input_num=128, output_num=256, stride=1)
        low_level5 = Convolution(input=low_level4, weight_name="low_level_w5", bias_name = "low_level_b5", input_num=256, output_num=256, stride=2)
        low_level6 = Convolution(input=low_level5, weight_name="low_level_w6", bias_name = "low_level_b6", input_num=256, output_num=512, stride=1)

        global_level1 = Convolution(input=low_level6, weight_name="global_level_w1", bias_name = "global_level_b1",  input_num=512, output_num=512, stride=2)
        global_level2 = Convolution(input=global_level1, weight_name="global_level_w2",bias_name = "global_level_b2", input_num=512, output_num=512, stride=1)
        global_level3 = Convolution(input=global_level2, weight_name="global_level_w3",bias_name = "global_level_b3", input_num=512, output_num=512, stride=2)
        global_level4 = Convolution(input=global_level3, weight_name="global_level_w4",bias_name = "global_level_b4", input_num=512, output_num=512, stride=1)
        global_level5 = Fully_connected(input= global_level4, weight_name="global_level_w5", bias_name="global_level_b5", input_num=7*7*512, output_num=1024)
        global_level6 = Fully_connected(input= global_level5, weight_name="global_level_w6", bias_name="global_level_b6",input_num=1024, output_num=512)
        global_level7 = Fully_connected(input = global_level6,weight_name="global_level_w7", bias_name="global_level_b7", input_num=512, output_num=256)

        mid_level1 = Convolution(input=low_level6, weight_name="mid_level_w1",bias_name="mid_level_b1", input_num=512, output_num=512, stride=1)
        mid_level2 = Convolution(input=mid_level1, weight_name="mid_level_w2",bias_name="mid_level_b2", input_num=512, output_num=256, stride=1)

        fusion_layer = Fusion(input1=mid_level2, input2=global_level7, weight_name="fusion_w", bias_name="fusion_b")

        colorization_network1 = Convolution(input=fusion_layer, weight_name="colorization_network1_w1",bias_name="colorization_network1_b1", input_num=256, output_num=128, stride=1)
        colorization_network2 = Upsample(input = colorization_network1,weight_name="colorization_network_w2", height_size = math.ceil(image_height_size/4), width_size = math.ceil(image_width_size/4), channel = 128)
        colorization_network3 = Convolution(input=colorization_network2, weight_name="colorization_network1_w3",bias_name="colorization_network1_b3", input_num=128, output_num=64, stride=1)
        colorization_network4 = Convolution(input=colorization_network3, weight_name="colorization_network1_w4",bias_name="colorization_network1_b4", input_num=64, output_num=64, stride=1)
        colorization_network5 = Upsample(input = colorization_network4,weight_name="colorization_network_w5", height_size = math.ceil(image_height_size/2), width_size = math.ceil(image_width_size/2), channel = 64)
        colorization_network6 = Convolution(input=colorization_network5, weight_name="colorization_network1_w6",bias_name="colorization_network1_b6", input_num=64, output_num=32, stride=1)
        colorization_network7 = Convolution(input=colorization_network6, weight_name="colorization_network1_w7",bias_name="colorization_network1_b7", input_num=32, output_num=2, stride=1)
        self.colorization_network8 = Upsample(input=colorization_network7, weight_name="colorization_network_w8", height_size=image_height_size, width_size=image_width_size, channel=2)

        colorization_loss = tf.losses.mean_squared_error(labels=self.Y_colorization, predictions=self.colorization_network8)

        classification_level1 = Fully_connected(input=global_level6, weight_name="classfication_level_w1", bias_name="classfication_level_b1", input_num=512, output_num=256)
        self.classification_level2 = Fully_connected(input=classification_level1, weight_name="classfication_level_w2", bias_name="classfication_level_b2", input_num=256, output_num=classification_num)

        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_classification_one_hot, logits=self.classification_level2))

        self.loss = colorization_loss - (loss_parameter * classification_loss)
        self.train = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(self.loss)

        loss_graph = tf.summary.scalar("loss",self.loss)
        self.merged_graph = tf.summary.merge_all()

    def Loss_graph(self, X, Y_colorization, Y_classification):
        return self.sess.run(self.merged_graph, feed_dict={self.X: X, self.Y_colorization: Y_colorization, self.Y_classification: Y_classification})

    def Loss(self, X, Y_colorization, Y_classification):
        return self.sess.run(self.loss, feed_dict={self.X: X, self.Y_colorization: Y_colorization, self.Y_classification: Y_classification})

    def Train(self, X, Y_colorization, Y_classification):
        return self.sess.run(self.train,feed_dict={self.X: X, self.Y_colorization : Y_colorization, self.Y_classification : Y_classification})

#varaible
epochs = 11 #11
batch_size = 2 #128
X, Y_colorization, Y_classification  = read_train_data()

#initialize
sess = tf.InteractiveSession()
model = Model(sess)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./board",sess.graph)

#train model
for epoch in range(epochs):
    loss = model.Loss(X, Y_colorization, Y_classification)
    print(loss)
    model.Train(X, Y_colorization, Y_classification)
    graph_loss = model.Loss_graph(X, Y_colorization, Y_classification)
    writer.add_summary(graph_loss,epoch)

saver.save(sess,'C:/Users/heojo/Desktop/Colorization/ckpt/my-model')
writer.close()
sess.close()

#tensorboard --logdir=/board
#localhost:6006
