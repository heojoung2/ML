import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

data_dir = './MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

x_input = tf.placeholder(tf.float32, shape=[None,784])
y_input = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x_input,[-1,28,28,1])

w1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1), name='w1')     #32개 5x5필터
b1 = tf.Variable(tf.constant(0.1, shape=[32]),name = 'b1')

h1 = tf.nn.conv2d(x_image,w1,strides=[1,1,1,1], padding="SAME") + b1
h1 = tf.nn.relu(h1)
h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

w2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=[64]),name='b2')

h2 = tf.nn.relu(tf.nn.conv2d(h1,w2,strides=[1,1,1,1], padding="SAME") + b2)
h2 = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

#fully connected
w3 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b3 =  tf.Variable(tf.constant(0.1, shape=[1024]))

h3 = tf.reshape(h2,[-1,7*7*64])
h3 = tf.nn.relu(tf.matmul(h3, w3) + b3)
h3 = tf.nn.dropout(h3,keep_prob)

w4 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[10]))

hypothesis = tf.matmul(h3, w4) + b4

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=hypothesis))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver({'w1':w1,'b1':b1,'w2':w2,'b2':b2})

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#train : 55000
#test : 10000
batch_size = 50
for i in range(0,55000,batch_size):
    batch = mnist.train.next_batch(batch_size)
    if i%100 == 0 :
        train_accuracy = accuracy.eval(feed_dict = {x_input:batch[0],y_input:batch[1], keep_prob:1.0})
        print('step',i,'training accuracy',train_accuracy)
    sess.run(train,feed_dict={x_input:batch[0], y_input:batch[1], keep_prob:0.5})

dir = os.path.dirname(os.path.realpath(__file__))
saver.save(sess, dir + '/ckpt1/my-model', global_step=i)

#testing
for i in range(0,10000,batch_size):
    batch = mnist.test.next_batch(batch_size)
    train_accuracy = accuracy.eval(feed_dict={x_input: batch[0], y_input: batch[1], keep_prob: 1.0})
    print('step',i,'testing accuracy', train_accuracy)
