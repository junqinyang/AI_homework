#添加层
#sklearn数据集

import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)#把输出层转换为一列10个神经元的形式
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)


#(输入的数据，输入数据的size，输出数据的size，激活函数)
def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):#这里我们激活函数默认选None，就不用tf内置的激活函数，简单的用一个线性函数
    #first define weight & bias
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#初值随机，大写表示这是矩阵,初始值用随机会比全0好很多
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#bias是个列表，只有1行，out_size列，初值不推荐为0，故+0.1

    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    #here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs',outputs)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction#把prediction变成全局变量(非必须)
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})#把xs放到prediction里面生成预测值y_pre
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))#将预测值和标准值比较
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#计算精度
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#define placeholder for inputs to network

keep_prob = tf.placeholder(tf.float32)#没有给它定义具体的形式，因为就是一个比例，表示被drop剩余的东西

xs = tf.placeholder(tf.float32,[None,64])#None表示不规定有多少个sample，但是每个sample有64个像素点，8x8=64
ys = tf.placeholder(tf.float32,[None,10])#每个sample有10个输出

#add output layer
l1 = add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)#softmax这个af适合用来分类

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))#交叉熵

#training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#这里的minimize()实际上就是loss

sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train',sess.graph)
test_writer = tf.summary.FileWriter('./test',sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(1000):
    #提取部分的x和y，分为一个个的batch，每次学习只学习部分batch，就更快些
    #batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})

    if i % 50 == 0:
        #printf(compute_accuracy(mnist.test.images,mnist.test.labels))
        
        #record loss
        train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i) 
        test_writer.add_summary(test_result,i)







'''
#numpy底子不行，重点介绍一下这个x_data。linspace函数，再切片，构成了这个x_data
#linspace构成的是一个一维列表，从-1到1，平均取300个值
#然后此处切片就是每个元素都单独切成一个列表，原本的大列表里就有300个子列表，每个子列表只有一个数
#因为是x值，所以这就定义了inputs的300行的值
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)#噪点，让数据更像真实数据。。。
y_data = np.square(x_data)-0.5+noise

#以上，x_data只有一个属性，故输入层只有一个神经元
#y_data也只有一个属性，故输出层只有一个神经元，隐藏层这里设有10个神经元，网络为三层结构

#下面定义placeholder给网络的输入付值
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#设置第一层输入层，选用relu作为激活函数
layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
#设定输出层
prediction = add_layer(layer1,10,1,activation_function=None)

#定义损失函数，计算预测值和标准值的差别
#reduce_mean就是tf里的求平均值
#reduce_sum就是tf里的求和，参数reduction_indices指定沿哪个维度sum
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    #training
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})#为什么要用placeholder呢？在某些网络结构，需要中途多次付值，所以这样比较好
    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))#每次要用到包含xs和ys的值的时候都要记得feed
'''

