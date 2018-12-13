#classify iris dataset from sklearn

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
movie = np.loadtxt(open('./movie.csv','r'),delimiter=',',skiprows=0)
score = np.loadtxt('./movieScore.txt')
score = score.astype(int)
score = LabelBinarizer().fit_transform(score)
X_train,X_test,y_train,y_test = train_test_split(movie,score,test_size=0.1)
'''
print(X_train.shape)
print(y_train.shape)
print(y_train)
'''



def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) 
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
            tf.summary.histogram(layer_name+'/Wx_plus_b',Wx_plus_b)
    
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

#compute accuracy
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#define placeholder
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,3])
    ys = tf.placeholder(tf.float32,[None,8])

#add hidden layer and output layer
with tf.name_scope('layer1'):
    l1 = add_layer(xs,3,10,'layer1',activation_function=tf.nn.tanh)
with tf.name_scope('prediction'):
    prediction = add_layer(l1,10,8,'prediction',activation_function=tf.nn.softmax)

#loss
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

#train step
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./",sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(1000):
    #training
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train})

    if i % 50 == 0:
        #print(sess.run(cross_entropy,feed_dict={xs:X_train,ys:y_train}))
        #print("  ")
        accu = compute_accuracy(X_test,y_test)
        print(accu)


