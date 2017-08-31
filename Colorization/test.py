#-*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import cv2

def Convolution(input, name, input_num, output_num, stride,reuse_flag=False):
    with tf.variable_scope(name, reuse=reuse_flag):
        weight = tf.get_variable("weight", shape=[3,3,input_num,output_num],initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", shape=[output_num], initializer=tf.contrib.layers.xavier_initializer())
        convolution = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding="SAME") + bias

        if name=="colorization_network7":
            sigmoid = tf.nn.sigmoid(convolution)
            return sigmoid

        relu = tf.nn.relu(convolution)
        return relu

def Fully_connected(input, name, input_num, output_num):
    with tf.variable_scope(name):
        weight = tf.get_variable("weight",shape=[input_num,output_num],initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("name",shape=[output_num],initializer=tf.contrib.layers.xavier_initializer())

        if name=="global_level5":
            input = tf.reshape(input,[-1,input_num])

        matmul_matrix = tf.matmul(input,weight)+bias
        relu = tf.nn.relu(matmul_matrix)
        return relu

def Fusion(input1,input2,weight_name,bias_name):
    col = int(input1.shape[1])
    row = int(input1.shape[2])
    input2 = tf.tile(input2,[1,col*row])
    input2 = tf.reshape(input2,[batch_size, col, row,256])
    input1 = tf.concat([input1,input2],3)
    weight = tf.get_variable(weight_name, shape=[1,1,512,256], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(bias_name, shape=[256], initializer=tf.contrib.layers.xavier_initializer())
    convolution = tf.nn.conv2d(input1, weight, strides=[1, 1, 1, 1], padding="SAME")+bias
    relu = tf.nn.relu(convolution)
    return relu

class Model:
    def __init__(self, sess,height,width):
        self.sess = sess
        self.width=width
        self.height=height
        self.build_network()

    def build_network(self):
        classification_num = 2 #205
        self.X = tf.placeholder(tf.float32, [batch_size,self.height,self.width,1])
        self.X2 = tf.placeholder(tf.float32, [batch_size, 224, 224,1])

        first_low_level1 = Convolution(input=self.X, name = "low_level1", input_num=1, output_num = 64, stride=2)
        first_low_level2 = Convolution(input=first_low_level1, name = "low_level2", input_num=64, output_num=128, stride=1)
        first_low_level3 = Convolution(input=first_low_level2, name = "low_level3", input_num=128, output_num=128, stride=2)
        first_low_level4 = Convolution(input=first_low_level3, name = "low_level4", input_num=128, output_num=256, stride=1)
        first_low_level5 = Convolution(input=first_low_level4, name = "low_level5", input_num=256, output_num=256, stride=2)
        first_low_level6 = Convolution(input=first_low_level5, name = "low_level6", input_num=256, output_num=512, stride=1)

        second_low_level1 = Convolution(input=self.X2, name = "low_level1", input_num=1, output_num = 64, stride=2,reuse_flag=True)
        second_low_level2 = Convolution(input=second_low_level1, name = "low_level2", input_num=64, output_num=128, stride=1,reuse_flag=True)
        second_low_level3 = Convolution(input=second_low_level2, name = "low_level3", input_num=128, output_num=128, stride=2,reuse_flag=True)
        second_low_level4 = Convolution(input=second_low_level3, name = "low_level4", input_num=128, output_num=256, stride=1,reuse_flag=True)
        second_low_level5 = Convolution(input=second_low_level4, name = "low_level5", input_num=256, output_num=256, stride=2,reuse_flag=True)
        second_low_level6 = Convolution(input=second_low_level5, name = "low_level6", input_num=256, output_num=512, stride=1,reuse_flag=True)

        mid_level1 = Convolution(input=first_low_level6, name="mid_level1", input_num=512, output_num=512, stride=1)
        mid_level2 = Convolution(input=mid_level1, name="mid_level2", input_num=512, output_num=256, stride=1)

        global_level1 = Convolution(input=second_low_level6, name = "global_level1",  input_num=512, output_num=512, stride=2)
        global_level2 = Convolution(input=global_level1, name = "global_level2", input_num=512, output_num=512, stride=1)
        global_level3 = Convolution(input=global_level2, name = "global_level3", input_num=512, output_num=512, stride=2)
        global_level4 = Convolution(input=global_level3, name = "global_level4", input_num=512, output_num=512, stride=1)
        global_level5 = Fully_connected(input= global_level4, name="global_level5", input_num=7*7*512, output_num=1024)
        global_level6 = Fully_connected(input= global_level5, name="global_level6",input_num=1024, output_num=512)
        global_level7 = Fully_connected(input = global_level6,name="global_level7", input_num=512, output_num=256)

        fusion_layer = Fusion(input1=mid_level2, input2=global_level7, weight_name="fusion_w", bias_name="fusion_b")

        colorization_network1 = Convolution(input=fusion_layer,name="colorization_network1", input_num=256, output_num=128, stride=1)
        colorization_network2 = tf.image.resize_images(colorization_network1, [math.ceil(self.height/ 4), math.ceil(self.width/ 4)])
        colorization_network3 = Convolution(input=colorization_network2, name="colorization_network3", input_num=128, output_num=64, stride=1)
        colorization_network4 = Convolution(input=colorization_network3, name="colorization_network4", input_num=64, output_num=64, stride=1)
        colorization_network5 = tf.image.resize_images(colorization_network4,[math.ceil(self.height / 2), math.ceil(self.width / 2)])
        colorization_network6 = Convolution(input=colorization_network5, name="colorization_network6", input_num=64, output_num=32, stride=1)
        colorization_network7 = Convolution(input=colorization_network6, name="colorization_network7", input_num=32, output_num=2, stride=1)
        self.colorization_network8 = tf.image.resize_images(colorization_network7,[self.height,self.width])

        classification_level1 = Fully_connected(input=global_level6,name="classification_level1", input_num=512, output_num=256)
        self.classification_level2 = Fully_connected(input=classification_level1,name="classification_level2", input_num=256, output_num=classification_num)

    def Predict_y_colorization(self, X, X2):
        return self.sess.run(self.colorization_network8, feed_dict={self.X: X, self.X2:X2})

    def Predict_y_classification(self, X, X2):
        return self.sess.run(tf.argmax(self.classification_level2,1), feed_dict={self.X: X, self.X2:X2})


batch_size = 1
image_size=224
image =cv2.imread('test_image.png')
image_height,image_width,image_channel =  image.shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
X=[image[:,:,:1]/255]

image2 = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
X2=[image2[:,:,:1]/255]

#initialize
sess = tf.InteractiveSession()
model = Model(sess,image_height,image_width)
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./ckpt/'))
#saver.restore(sess, "./ckpt/my-model-4")

colorization_result = model.Predict_y_colorization(X,X2)
classification_result = model.Predict_y_classification(X,X2)

L=np.reshape(image[:,:,:1],[image_height,image_width,1])
lab = np.concatenate((L,colorization_result[0]), axis=2)
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imshow('result : '+str(classification_result[0]),result)
cv2.waitKey(0)

sess.close()