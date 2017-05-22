# -*- encoding:utf-8 -*-

import h5py
import numpy as np
import tensorflow as tf

def batch_function(cnt):    #batch
    x_batch=[]
    y_batch=[]

    for i in range(cnt,cnt+batch_size):
        x_batch.append(np.reshape(images[i],(2704,))/255)
        y_batch.append(np.eye(14)[labels[i]])

    return x_batch, y_batch

x_input = tf.placeholder(tf.float32, shape=[None,2704])
y_input = tf.placeholder(tf.float32, shape=[None,14])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x_input,[-1,52,52,1])

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
w4 = tf.get_variable("w4", shape=[7*7*128, 1024],initializer=tf.contrib.layers.xavier_initializer())
b4 =  tf.Variable(tf.constant(0.1, shape=[1024]))

h4 = tf.reshape(h3,[-1,7*7*128])
h4 = tf.nn.relu(tf.matmul(h4, w4) + b4)
h4 = tf.nn.dropout(h4,keep_prob)

w5 = tf.get_variable("w5", shape=[1024, 14],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.constant(0.1, shape=[14]))

hypothesis = tf.matmul(h4, w5) + b5

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=hypothesis))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

sess = tf.InteractiveSession()
saver.restore(sess, tf.train.latest_checkpoint('./ckpt2/'))

with h5py.File('kalph_test.hf', 'r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])

batch_size=392
for i in range(0,3920,batch_size):
    x,y = batch_function(i)
    _accuracy = sess.run(accuracy, feed_dict={x_input:x, y_input:y, keep_prob:1.0})
    print('step',i, 'accuracy', _accuracy*100)
