# -*- encoding:utf-8 -*-

import h5py
import numpy as np
import tensorflow as tf
import os
import cv2

def batch_function(cnt):    #batch
    x_batch=[]
    y_batch=[]

    for i in range(cnt,cnt+batch_size):
        x_batch.append(np.reshape(images[i],(2704,))/255)
        y_batch.append(np.eye(14)[labels[i]])

    return x_batch, y_batch

def noise(cnt):    #노이즈
    x_batch = []

    im = np.zeros((52, 52), np.uint8)
    noise = cv2.randu(im, (0), (10))

    for i in range(cnt, cnt + batch_size):
        x_batch.append(np.reshape((images[i]+noise), (2704,)) / 255)

    return x_batch


def rotation(cnt,angle):    #회전
    x_batch = []
    for i in range(cnt, cnt + batch_size):
        M = cv2.getRotationMatrix2D((26, 26), angle, 1)
        img = cv2.warpAffine(images[i], M, (52, 52))

        x_batch.append(np.reshape(img, (2704,)) / 255)

    return x_batch

def translation(cnt,x,y):  #이동
    x_batch = []

    for i in range(cnt,cnt+batch_size):
        M = np.float32([[1,0,x],[0,1,y]])
        img = cv2.warpAffine(images[i],M,(52,52))

        x_batch.append(np.reshape(img,(2704,))/255)
    return x_batch

def expansion(cnt):     #확대
    x_batch = []

    for i in range(cnt, cnt + batch_size):
        expansion = cv2.resize(images[i], None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        img = np.zeros((52, 52), np.uint8)

        for j in range(0, 52):
            for k in range(0, 52):
                img[j][k] = expansion[j + 5][k + 5]

        x_batch.append(np.reshape(img, (2704,)) / 255)
    return x_batch

def reduction(cnt):     #축소
    x_batch = []

    for i in range(cnt, cnt + batch_size):
        reduction = cv2.resize(images[i], None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
        img = np.zeros((52, 52), np.uint8)

        for j in range(5, 47):
            for k in range(5, 47):
                img[j][k] = reduction[j - 5][k - 5]

        x_batch.append(np.reshape(img, (2704,)) / 255)
    return x_batch

def mopology_dilation(cnt):     #팽창
    x_batch = []
    kernel = np.ones((2, 2), np.uint8)

    for i in range(cnt, cnt + batch_size):
        dilation_img = cv2.dilate(images[i],kernel,iterations=1)
        x_batch.append(np.reshape(dilation_img, (2704,)) / 255)

    return x_batch

def mopology_closing(cnt):     #닫기
    x_batch = []
    kernel = np.ones((2, 2), np.uint8)

    for i in range(cnt, cnt + batch_size):
        closing_img = cv2.morphologyEx(images[i], cv2.MORPH_CLOSE, kernel)
        x_batch.append(np.reshape(closing_img, (2704,)) / 255)

    return x_batch


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

saver = tf.train.Saver({'w1':w1,'b1':b1,'w2':w2,'b2':b2})
saver2 = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver.restore(sess, tf.train.latest_checkpoint('./ckpt1/'))


with h5py.File('kalph_train.hf', 'r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])

num_imgs, rows, cols = images.shape     #num_imgs:19600, rows:52, cols:52

#train : 19600
#test : 3920
batch_size=50
for epoch in range(3):
    for i in range(0,19600,batch_size):
        x,y = batch_function(i)
        sess.run(train, feed_dict={x_input:x, y_input:y, keep_prob:0.5})

        x_n = noise(i)  #노이즈
        sess.run(train,feed_dict={x_input:x_n, y_input:y, keep_prob:0.5})

        for angle in range(-30,35,5):  #회전
            x_r = rotation(i,angle)
            sess.run(train, feed_dict={x_input: x_r, y_input: y, keep_prob: 0.5})

        for move_x in range(-3,4):       #이동
            for move_y in range(-3,4):
                if not (move_x==0 and move_y==0):
                    x_t = translation(i,move_x,move_y)
                    sess.run(train, feed_dict={x_input: x_t, y_input: y, keep_prob: 0.5})

        x_e = expansion(i)      #확대
        sess.run(train, feed_dict={x_input: x_e, y_input: y, keep_prob: 0.5})

        x_r = reduction(i)      #축소
        sess.run(train, feed_dict={x_input: x_r, y_input: y, keep_prob: 0.5})

        x_m_d = mopology_dilation(i)    #모폴로지 팽창
        sess.run(train, feed_dict={x_input:x_m_d, y_input:y, keep_prob:0.5})

        x_m_c = mopology_closing(i)     #모폴로지 닫기
        sess.run(train, feed_dict={x_input: x_m_c, y_input: y, keep_prob: 0.5})

        #_accuracy = sess.run(accuracy, feed_dict={x_input: x, y_input: y, keep_prob: 1.0})
        print('epoch',epoch,'step',i)

dir = os.path.dirname(os.path.realpath(__file__))
saver2.save(sess, dir + '/ckpt2/my-model', global_step=epoch)


#test
with h5py.File('kalph_test.hf', 'r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])

batch_size=392
for i in range(0,3920,batch_size):
    x,y = batch_function(i)
    _accuracy = sess.run(accuracy, feed_dict={x_input:x, y_input:y, keep_prob:1.0})
    print('step',i, 'accuracy', _accuracy*100)
