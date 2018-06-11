
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#定义一个placeholder
y = tf.placeholder(tf.float32,[None,10])

#载入模型
with tf.gfile.FastGFile('./models/tfmodel.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    output = sess.graph.get_tensor_by_name('output:0')
    #结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))#argmax返回一维张量中最大的值所在的位置
    #求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(sess.run(accuracy,feed_dict={'x-input:0':mnist.test.images,y:mnist.test.labels}))


# In[ ]:



