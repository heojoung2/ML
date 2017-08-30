#-*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import cv2

def Fusion(input1,input2,name):
    col = int(input1.shape[1])
    row = int(input1.shape[2])
    input2 = tf.tile(input2,[1,col*row])
    input2 = tf.reshape(input2,[batch_size, col, row,256])
    input1 = tf.concat([input1,input2],3)
    fusion = slim.conv2d(input1, 256, [3, 3], scope=name)
    return fusion

class Model:
    def __init__(self, sess,height,width):
        self.sess = sess
        self.width=width
        self.height=height
        self.build_network()

    def build_network(self):
        classification_num = 2 #205
        loss_parameter = 1/300
        learning_rate = 0.001

        self.X = tf.placeholder(tf.float32, [batch_size,self.height,self.width])
        X = tf.reshape(self.X, [batch_size,self.height,self.width,1])

        self.X2 = tf.placeholder(tf.float32, [batch_size, 224, 224])
        X2 = tf.reshape(self.X2, [batch_size, 224, 224, 1])

        first_low_level = slim.stack(X, slim.conv2d, [(64, [3, 3], 2), (128, [3, 3], 1), (128, [3, 3], 2), (256, [3, 3], 1), (256,[3,3], 2), (512,[3,3],1)],scope='low_level')

        second_low_level1 = slim.conv2d(X2, 64, [3, 3], stride=2, scope='low_level/low_level_1',reuse=True)
        second_low_level2 = slim.conv2d(second_low_level1, 128, [3, 3], scope='low_level/low_level_2',reuse=True)
        second_low_level3 = slim.conv2d(second_low_level2, 128, [3, 3], stride=2, scope='low_level/low_level_3',reuse=True)
        second_low_level4 = slim.conv2d(second_low_level3, 256, [3, 3], scope='low_level/low_level_4',reuse=True)
        second_low_level5 = slim.conv2d(second_low_level4, 256, [3, 3], stride=2, scope='low_level/low_level_5',reuse=True)
        second_low_level6 = slim.conv2d(second_low_level5, 512, [3, 3], scope='low_level/low_level_6', reuse=True)

        mid_level = slim.stack(first_low_level, slim.conv2d, [(512,[3,3]), (256,[3,3])],scope='mid_level')

        global_level1 = slim.stack(second_low_level6, slim.conv2d, [(512, [3, 3], 2), (512, [3, 3], 1), (512, [3, 3], 2), (512, [3, 3], 1)], scope='global_level1')
        global_level1 = tf.reshape(global_level1, [-1, 7*7*512])
        global_level2 = slim.fully_connected(global_level1, 1024, scope='global_level2')
        global_level3 = slim.fully_connected(global_level2, 512, scope='global_level3')
        global_level4 = slim.fully_connected(global_level3, 256, scope='global_level4')

        fusion_layer = Fusion(input1=mid_level, input2=global_level4, name="fusion")

        colorization_network1 = slim.conv2d(fusion_layer,128,[3,3],scope="colorization_network1")
        colorization_network2 = tf.image.resize_images(colorization_network1, [math.ceil(self.height / 4), math.ceil(self.width / 4)])
        colorization_network3 = slim.stack(colorization_network2, slim.conv2d, [(64,[3,3]), (64,[3,3])],scope='colorization_network3')
        colorization_network4 = tf.image.resize_images(colorization_network3, [math.ceil(self.height / 2),math.ceil(self.width / 2)])
        colorization_network5 = slim.conv2d(colorization_network4, 32, [3, 3], scope="colorization_network5")

        colorization_network6 = slim.conv2d(colorization_network5, classification_num, [3, 3], scope="colorization_network6", activation_fn=tf.nn.sigmoid)
        self.colorization_network7 = tf.image.resize_images(colorization_network6, [self.height,self.width])

        classification_level1 = slim.fully_connected(global_level3, 256, scope='classficatio_level1')
        self.classification_level2 = slim.fully_connected(classification_level1, classification_num, scope='classficatio_level2')

    def Predict_y_colorization(self, X, X2):
        return self.sess.run(self.colorization_network7, feed_dict={self.X: X, self.X2:X2})

    def Predict_y_classification(self, X, X2):
        return self.sess.run(tf.argmax(self.classification_level2,1), feed_dict={self.X: X, self.X2:X2})


batch_size = 1
image_size=224
image =cv2.imread('test_image.png',cv2.IMREAD_GRAYSCALE)
image_height,image_width =  image.shape
X=[image]

image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
X2=[image]

#initialize
sess = tf.InteractiveSession()
model = Model(sess,image_height,image_width)
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./ckpt/'))
#saver.restore(sess, "./ckpt/my-model-4")

colorization_result = model.Predict_y_colorization(X,X2)
classification_result = model.Predict_y_classification(X,X2)

X=np.reshape(X,[image_height,image_width,1])
X=X.astype(np.float32)
X=X*100
X=X/255
X=X.astype(np.uint8)
lab = np.concatenate((X,colorization_result[0]), axis=2)
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imshow('result : '+str(classification_result[0]),result)
cv2.waitKey(0)

sess.close()