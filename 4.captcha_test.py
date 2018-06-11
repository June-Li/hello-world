
# coding: utf-8

# In[1]:

import os
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt  
import tensorflow.contrib.slim as slim
get_ipython().magic('matplotlib inline')


# In[2]:

# 不同字符数量
CHAR_SET_LEN = 10

BATCH_SIZE = 1
# tfrecord文件存放路径
TFRECORD_FILE = "V:/python-ai-3/day3/captcha/test.tfrecords"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])  

# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # 没有经过预处理的灰度图
    image_raw = tf.reshape(image, [224, 224])
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3


# In[3]:

# 获取图片数据和标签
image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

#使用shuffle_batch可以随机打乱
image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, image_raw, label0, label1, label2, label3], batch_size = BATCH_SIZE,
        capacity = 50000, min_after_dequeue=10000, num_threads=1)


def alexnet(inputs, is_training=True):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                         activation_fn=tf.nn.relu,
                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                         biases_initializer=tf.constant_initializer(0.1),
                         weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
        net = slim.max_pool2d(net, [3, 3], scope='pool1')
        net = slim.conv2d(net, 192, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [3, 3], scope='pool2')
        net = slim.conv2d(net, 384, [3, 3], scope='conv3')
        net = slim.conv2d(net, 384, [3, 3], scope='conv4')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [3, 3], scope='pool5')
        
        # 数据扁平化
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, scope='fc6')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
        net = slim.fully_connected(net, 1024, scope='fc7')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
        
        net0 = slim.fully_connected(net, CHAR_SET_LEN, activation_fn=None, scope='fc8_0')
        net1 = slim.fully_connected(net, CHAR_SET_LEN, activation_fn=None, scope='fc8_1')
        net2 = slim.fully_connected(net, CHAR_SET_LEN, activation_fn=None, scope='fc8_2')
        net3 = slim.fully_connected(net, CHAR_SET_LEN, activation_fn=None, scope='fc8_3')

    return net0,net1,net2,net3

with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits0,logits1,logits2,logits3 = alexnet(X,False)

    # 预测值
    predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])  
    predict0 = tf.argmax(predict0, 1)  

    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])  
    predict1 = tf.argmax(predict1, 1)  

    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])  
    predict2 = tf.argmax(predict2, 1)  

    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])  
    predict3 = tf.argmax(predict3, 1)  

    # 初始化
    sess.run(tf.global_variables_initializer())
    # 载入训练好的模型
    saver = tf.train.Saver()
    saver.restore(sess,'./captcha/models/crack_captcha.model-6000')

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        # 获取一个批次的数据和标签
        b_image, b_image_raw, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, 
                                                                    image_raw_batch, 
                                                                    label_batch0, 
                                                                    label_batch1, 
                                                                    label_batch2, 
                                                                    label_batch3])
        # 显示图片
        plt.imshow(b_image_raw[0])
        plt.axis('off')
        plt.show()
        # 打印标签
        print('label:',b_label0, b_label1 ,b_label2 ,b_label3)
        # 预测
        label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3], feed_dict={x: b_image})
        # 打印预测值
        print('predict:',label0,label1,label2,label3) 
                
    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)

