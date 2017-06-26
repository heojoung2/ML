# -*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import cv2

pos_train_path = 'C:/Users/heo/Desktop/pos/pos_train/'
pos_test_path = 'C:/Users/heo/Desktop/pos/pos_test/'
neg_train_path = 'C:/Users/heo/Desktop/neg/neg_train/'
neg_test_path = 'C:/Users/heo/Desktop//neg/neg_test/'

batch_size=100

def read_image(something):
    image=[]
    label=[]

    if something == 'train':
        pos_train_images = os.listdir(pos_train_path)
        neg_train_images = os.listdir(neg_train_path)

        for name in pos_train_images:
            im = cv2.imread(pos_train_path + name, 0)
            im = cv2.resize(im,(70,134),interpolation=cv2.INTER_AREA)
            image.append(im/255)
            label.append([0,1])

        for name in neg_train_images:
            im = cv2.imread(neg_train_path + name, 0)
            im = cv2.resize(im,(70,134),interpolation=cv2.INTER_AREA)
            image.append(im/255)
            label.append([1,0])

    elif something=='test':
        pos_test_images = os.listdir(pos_test_path)
        neg_test_images = os.listdir(neg_test_path)

        for name in pos_test_images:
            im = cv2.imread(pos_test_path + name, 0)
            im = cv2.resize(im,(70,134),interpolation=cv2.INTER_AREA)
            image.append(im/255)
            label.append([0,1])

        for name in neg_test_images:
            im = cv2.imread(neg_test_path + name, 0)
            im = cv2.resize(im,(70,134),interpolation=cv2.INTER_AREA)
            image.append(im/255)
            label.append([1,0])

    return np.array(image), np.array(label)

test_image, test_label = read_image('test')

x_input = tf.placeholder(tf.float32, shape=[None,134,70])
y_input = tf.placeholder(tf.float32, shape=[None,2])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x_input,[-1,134,70,1])  #55x40

w1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1), name='w1')     #32개 5x5필터
b1 = tf.Variable(tf.constant(0.1, shape=[32]),name = 'b1')

h1 = tf.nn.conv2d(x_image,w1,strides=[1,1,1,1], padding="SAME") + b1
h1 = tf.nn.relu(h1)
h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
h1 = tf.nn.dropout(h1,keep_prob)

w2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=[64]),name='b2')

h2 = tf.nn.conv2d(h1,w2,strides=[1,1,1,1], padding="SAME") + b2
h2 = tf.nn.relu(h2)
h2 = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
h2 = tf.nn.dropout(h2,keep_prob)

w3 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01))
b3 = tf.Variable(tf.constant(0.1, shape=[128]))

h3 = tf.nn.conv2d(h2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3
h3 = tf.nn.relu(h3)
h3 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
h3 = tf.nn.dropout(h3, keep_prob=keep_prob)

#fully connected
w4 = tf.get_variable("w4", shape=[17*9*128, 1024],initializer=tf.contrib.layers.xavier_initializer())
b4 =  tf.Variable(tf.constant(0.1, shape=[1024]))

h4 = tf.reshape(h3,[-1,17*9*128])
h4 = tf.nn.relu(tf.matmul(h4, w4) + b4)
h4 = tf.nn.dropout(h4,keep_prob)

w5 = tf.get_variable("w5", shape=[1024, 2],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.constant(0.1, shape=[2]))

hypothesis = tf.matmul(h4, w5) + b5

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=hypothesis))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

test_batch_x, test_batch_y = tf.train.batch([test_image,test_label],batch_size=batch_size,enqueue_many=True,capacity=1200)

saver = tf.train.Saver()

sess = tf.InteractiveSession()
saver.restore(sess, tf.train.latest_checkpoint('./ckpt/'))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(0,len(test_image),batch_size):
    x,y = sess.run([test_batch_x,test_batch_y])
    _accuracy = sess.run(accuracy, feed_dict={x_input: x, y_input: y, keep_prob: 1.0})
    print('step', i, 'accuracy',_accuracy)
