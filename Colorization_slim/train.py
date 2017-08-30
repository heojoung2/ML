#-*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py

def read_train_data():
    with h5py.File('train_data.hf', 'r') as hf:
        X = np.array(hf["X"])
        Y_colorization = np.array(hf["Y_colorization"])
        Y_classification = np.array(hf["Y_classification"])

    return X, Y_colorization, Y_classification

def Fusion(input1,input2, name):
    input2 = tf.tile(input2,[1,28*28])
    input2 = tf.reshape(input2,[batch_size,28,28,256])
    input1 = tf.concat([input1,input2],3)
    fusion = slim.conv2d(input1, 256, [3, 3], scope=name)
    return fusion

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

        first_low_level = slim.stack(X, slim.conv2d, [(64, [3, 3], 2), (128, [3, 3], 1), (128, [3, 3], 2), (256, [3, 3], 1), (256,[3,3], 2), (512,[3,3],1)],scope='low_level')

        second_low_level1 = slim.conv2d(X, 64, [3, 3], stride=2, scope='low_level/low_level_1',reuse=True)
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
        colorization_network2 = slim.conv2d_transpose(colorization_network1, 128, 2, stride=2, scope="colorization_network2")
        colorization_network3 = slim.stack(colorization_network2, slim.conv2d, [(64,[3,3]), (64,[3,3])],scope='colorization_network3')
        colorization_network4 = slim.conv2d_transpose(colorization_network3, 64, 2, stride=2, scope="colorization_network4")
        colorization_network5 = slim.conv2d(colorization_network4, 32, [3, 3], scope="colorization_network5")

        colorization_network6 = slim.conv2d(colorization_network5, classification_num, [3, 3], scope="colorization_network6", activation_fn=tf.nn.sigmoid)
        self.colorization_network7 = slim.conv2d_transpose(colorization_network6, 2, 3, stride=2, scope="colorization_network7")

        colorization_loss = tf.losses.mean_squared_error(labels=self.Y_colorization, predictions=self.colorization_network7)

        classification_level1 = slim.fully_connected(global_level3, 256, scope='classficatio_level1')
        self.classification_level2 = slim.fully_connected(classification_level1, classification_num, scope='classficatio_level2')

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
epochs = 1 #11
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

saver.save(sess,'C:/Users/heojo/Desktop/Colorization_slim/ckpt/my-model')
writer.close()
sess.close()

#tensorboard --logdir=/board
#localhost:6006
