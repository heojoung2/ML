#-*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import cv2
import os

def colorization_arg_scope(weight_decay=0.00004,
                           use_batch_norm=True,
                           batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        #  epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        #  collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        #  collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
     # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params) as sc:
            return sc

def Fusion(input1, input2, name):
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
        classification_num = 205

        self.X = tf.placeholder(tf.float32, [batch_size,self.height,self.width,1])
        self.X2 = tf.placeholder(tf.float32, [batch_size, 224, 224,1])

        with slim.arg_scope(colorization_arg_scope()):
            with slim.arg_scope([slim.batch_norm], is_training=False):
                first_low_level = slim.stack(self.X, slim.conv2d, [(64, [3, 3], 2), (128, [3, 3], 1), (128, [3, 3], 2), (256, [3, 3], 1), (256,[3,3], 2), (512,[3,3],1)],scope='low_level')

                second_low_level1 = slim.conv2d(self.X2, 64, [3, 3], stride=2, scope='low_level/low_level_1',reuse=True)
                second_low_level2 = slim.conv2d(second_low_level1, 128, [3, 3], scope='low_level/low_level_2',reuse=True)
                second_low_level3 = slim.conv2d(second_low_level2, 128, [3, 3],stride=2, scope='low_level/low_level_3',reuse=True)
                second_low_level4 = slim.conv2d(second_low_level3, 256, [3, 3], scope='low_level/low_level_4',reuse=True)
                second_low_level5 = slim.conv2d(second_low_level4, 256, [3, 3], stride=2, scope='low_level/low_level_5',reuse=True)
                second_low_level6 = slim.conv2d(second_low_level5, 512, [3, 3], scope='low_level/low_level_6', reuse=True)

                mid_level = slim.stack(first_low_level, slim.conv2d, [(512,[3,3]), (256,[3,3])],scope='mid_level' )

                global_level1 = slim.stack(second_low_level6, slim.conv2d, [(512, [3, 3], 2), (512, [3, 3], 1), (512, [3, 3], 2), (512, [3, 3], 1)], scope='global_level1')
                global_level1 = slim.flatten(global_level1)
                global_level2 = slim.fully_connected(global_level1, 1024, scope='global_level2')
                global_level3 = slim.fully_connected(global_level2, 512, scope='global_level3')
                global_level4 = slim.fully_connected(global_level3, 256, scope='global_level4')

                fusion_layer = Fusion(input1=mid_level, input2=global_level4, name="fusion")

                colorization_network1 = slim.conv2d(fusion_layer,128,[3,3],scope="colorization_network1")
                colorization_network2 = tf.image.resize_images(colorization_network1, [math.ceil(self.height / 4), math.ceil(self.width / 4)])
                colorization_network3 = slim.stack(colorization_network2, slim.conv2d, [(64,[3,3]), (64,[3,3])],scope='colorization_network3')
                colorization_network4 = tf.image.resize_images(colorization_network3, [math.ceil(self.height / 2),math.ceil(self.width / 2)])
                colorization_network5 = slim.conv2d(colorization_network4, 32, [3, 3],scope="colorization_network5")

                colorization_network6 = slim.conv2d(colorization_network5, 2, [3, 3], scope="colorization_network6", activation_fn=tf.nn.sigmoid)
                self.colorization_network7 = tf.image.resize_images(colorization_network6, [self.height,self.width])

                classification_level1 = slim.fully_connected(global_level3, 256, scope='classfication_level1')
                self.classification_level2 = slim.fully_connected(classification_level1, classification_num, scope='classfication_level2')

    def Predict_y_colorization(self, X, X2):
        return self.sess.run(self.colorization_network7, feed_dict={self.X: X, self.X2:X2})

    def Predict_y_classification(self, X, X2):
        return self.sess.run(tf.argmax(self.classification_level2,1), feed_dict={self.X: X, self.X2:X2})

batch_size = 1
image_size=224
image =cv2.imread('C:/users/heojo/Desktop/Colorization_slim/test_image.PNG')

image_height,image_width,image_channel =  image.shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
X=[image[:,:,:1]/255]

image2 = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
X2=[image2[:,:,:1]/255]

#initialize
sess = tf.InteractiveSession()
model = Model(sess,image_height,image_width)
saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('./ckpt/'))
saver.restore(sess, "./ckpt/my-model-1")

colorization_result = model.Predict_y_colorization(X,X2)
classification_result = model.Predict_y_classification(X,X2)

L=np.reshape(image[:,:,:1],[image_height,image_width,1])
ab = (colorization_result[0]*255).astype(np.uint8)
lab = np.concatenate( (L, ab), axis=2)
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

category = os.listdir("D:/images256/")
category_name = [j for i,j in enumerate(category)]

cv2.imshow('gray',L)
cv2.imshow('result : '+ category_name[classification_result[0]],result)
cv2.waitKey(0)

sess.close()