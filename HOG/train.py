# -*- encoding:utf-8 -*-
import cv2
import numpy as np
import copy
import math
import os
import tensorflow as tf

pos_train_path = 'C:/Users/heo/Desktop/pos/pos_train/'
pos_test_path = 'C:/Users/heo/Desktop/pos/pos_test/'
neg_train_path = 'C:/Users/heo/Desktop/neg/neg_train/'
neg_test_path = 'C:/Users/heo/Desktop//neg/neg_test/'

batch_size = 20

def hog_function(image):

    im = cv2.imread(image,0)
    im = cv2.resize(im,(70,134),interpolation=cv2.INTER_AREA)
    rows, cols = im.shape      #134, 70
    im=im.astype(np.float32)    #unsigned -> float

    lx = copy.copy(im)
    ly = copy.copy(im)

    for i in range(rows-2):     #y미분
        ly[i,:]=(im[i,:]-im[i+2,:])

    for i in range(cols-2):     #x미분
        lx[:,i]=(im[:,i]- im[:,i+2])

    angle = copy.copy(im)
    magnitue = copy.copy(im)

    for i in range(rows):
        for j in range(cols):
            if ly[i][j]==0:
                angle[i][j] = 0
            else:
                angle[i][j] = math.degrees(math.atan(lx[i][j]/ly[i][j]))+90
            magnitue[i][j]=math.sqrt(math.pow(lx[i][j],2)+math.pow(ly[i][j],2))

    feature=[]      #초기화된 피쳐 벡터
    patch=0
    cells=4
    bin=9
    block=16
    shift=8

    for i in range(int(rows/shift)):         #블록
        for j in range(int(cols/shift)):
            block_feature=[]
            patch+=1

            mag_patch = magnitue[i * shift: i * shift + block, j * shift: j * shift + block]
            ang_patch = angle[i * shift: i * shift + block, j * shift: j * shift + block]

            if len(mag_patch[0])!=block and len(mag_patch)!=block:
                mag_patch = magnitue[rows-block:rows, cols - block: cols]
                ang_patch = angle[rows-block:rows, cols - block: cols]

            elif len(mag_patch[0])!=block:
                mag_patch = magnitue[i * shift: i * shift + block, cols-block: cols]
                ang_patch = angle[i * shift: i * shift + block, cols-block: cols]
            elif len(mag_patch)!=block:
                mag_patch = magnitue[rows-block:rows, j * shift: j * shift + block]
                ang_patch = angle[rows-block:rows, j * shift: j * shift + block]

            for x in range(0,2):        #셀
                for y in range(0,2):

                    angleA = ang_patch[x*8:x*8+8,y*8:y*8+8]
                    magA = mag_patch[x*8:x*8+8,y*8:y*8+8]
                    histr = np.zeros(bin)

                    for p in range(0,8):        #픽셀
                        for q in range(0,8):
                            alpha = angleA[p][q]

                            cnt=0
                            for degree in range(10,170,20): #10,30,50,70,90,110,130,150
                                if alpha>degree and alpha <=(degree+20):
                                    histr[cnt] = histr[cnt] + magA[p,q]*(degree+20-alpha)/20
                                    histr[cnt+1] = histr[cnt+1] + magA[p,q]*(alpha-degree)/20
                                    cnt+=1
                            if alpha>170 and alpha<=180:
                                histr[8] = histr[8] + magA[p, q] * (180 - alpha) / 20
                                histr[0] = histr[0] + magA[p, q] * (alpha - 170) / 20
                            elif alpha>=0 and alpha<=10:
                                histr[0] = histr[0] + magA[p, q] * (alpha + 10) / 20
                                histr[8] = histr[8] + magA[p, q] * (10 - alpha) / 20

                    block_feature.append(histr)
#            block_feature=(block_feature / np.sqrt(math.pow(np.linalg.norm(block_feature), 2) + 0.01))  # 노말라이즈
            feature.append(block_feature)

#    feature = feature/np.sqrt(math.pow(np.linalg.norm(feature),2)+0.001)

    feature=np.array(feature)
    feature = feature.reshape(patch*cells*bin,1)
    feature = np.squeeze(feature)

#    for i in range(len(feature)):
#        if feature[i]>0.2:
#            feature[i]=0.2

    feature = feature/np.sqrt(math.pow(np.linalg.norm(feature),2)+0.001)*100
    return feature  #4680

def read_image(something):
    image=[]
    label=[]

    if something == 'train':
        pos_train_images = os.listdir(pos_train_path)
        neg_train_images = os.listdir(neg_train_path)

        for name in pos_train_images:
            im = hog_function(pos_train_path + name)
            image.append(im)
            label.append([0,1])

        for name in neg_train_images:
            im = hog_function(neg_train_path + name)
            image.append(im)
            label.append([1,0])

    elif something=='test':
        pos_test_images = os.listdir(pos_test_path)
        neg_test_images = os.listdir(neg_test_path)

        for name in pos_test_images:
            im = hog_function(pos_test_path + name)
            image.append(im)
            label.append([0,1])

        for name in neg_test_images:
            im = hog_function(neg_test_path + name)
            image.append(im)
            label.append([1,0])

    return np.array(image), np.array(label)

train_image, train_label = read_image('train')  # 134x70
#test_image, test_label = read_image('test')

x_input = tf.placeholder(tf.float32, shape=[None,4608])
y_input = tf.placeholder(tf.float32, shape=[None,2])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x_input,[-1,128,36,1])

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
w4 = tf.get_variable("w4", shape=[16*5*128, 1024],initializer=tf.contrib.layers.xavier_initializer())
b4 =  tf.Variable(tf.constant(0.1, shape=[1024]))

h4 = tf.reshape(h3,[-1,16*5*128])
h4 = tf.nn.relu(tf.matmul(h4, w4) + b4)
h4 = tf.nn.dropout(h4,keep_prob)

w5 = tf.get_variable("w5", shape=[1024, 2],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.constant(0.1, shape=[2]))

hypothesis = tf.matmul(h4, w5) + b5

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=hypothesis))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_batch_x, train_batch_y = tf.train.shuffle_batch([train_image,train_label],batch_size=batch_size,capacity=2000,min_after_dequeue=1000,enqueue_many=True)
#test_batch_x, test_batch_y = tf.train.batch([test_image,test_label],batch_size=batch_size,enqueue_many=True,capacity=1200)

saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for epoch in range(20):
    for i in range(0,len(train_image),batch_size):
        x,y = sess.run([train_batch_x,train_batch_y])
        sess.run(train, feed_dict={x_input:x, y_input:y, keep_prob:1.0})
        _accuracy = sess.run(accuracy, feed_dict={x_input: x, y_input: y, keep_prob: 1.0})

        print(epoch, i, _accuracy)

dir = os.path.dirname(os.path.realpath(__file__))
saver.save(sess, dir + '/ckpt/my-model', global_step=epoch)
'''
for i in range(0,len(test_image),batch_size):
    x,y = sess.run([test_batch_x,test_batch_y])
    _accuracy = sess.run(accuracy, feed_dict={x_input: x, y_input: y, keep_prob: 1.0})
    print('step', i, 'accuracy',_accuracy)
'''
