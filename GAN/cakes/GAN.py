import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
cakes_np=np.load('E:/Tensorflow/GAN/cakes/cake/nparray_cakes.npy')

# create tensorflow generator model
def generator():
    # input varies from 0 to 1 annd the output too
    s1,s2,s3,s4=4,8,64,128
    c1,c2,c3,c4=128,64,32,16
    inp1=tf.placeholder(dtype=tf.float16,shape=[None,100],name='input_gen')
    #dense
    with tf.variable_scope('generator'):
        h1=tf.layers.dense(inp1,s1*s1*d1,name='denseh1')
        reshape=tf.reshape(h1,[s1,s1,d1])
        #4x4x128
        convT1=tf.layers.conv2d_transpose(inp1,filters=c1,kernel_size=5, strides=(2, 2), padding='valid',name='ConvT1')
        bn1=tf.layers.batch_normalization(h2convT,name='Bnorm1')
        relu1=tf.nn.relu(bn1,name='Relu1')
        #8x8x64
        convT2=tf.layers.conv2d_transpose(relu1,filters=c2,kernel_size=5, strides=(2, 2), padding='valid',name='ConvT2')
        bn2=tf.layers.batch_normalization(convT2,name='Bnorm2')
        relu2=tf.nn.relu(bn2,name='Relu2')
        #32x32x32
        convT3=tf.layers.conv2d_transpose(relu2,filters=c3,kernel_size=5, strides=(2, 2), padding='valid',name='ConvT3')
        bn3=tf.layers.batch_normalization(convT3,name='Bnorm3')
        relu3=tf.nn.relu(bn3,name='Relu3')
        #64x64x16
        convT4=tf.layers.conv2d_transpose(relu3,filters=c4,kernel_size=5, strides=(2, 2), padding='valid',name='ConvT4')
        bn4=tf.layers.batch_normalization(convT4,name='Bnorm3')
        relu4=tf.nn.relu(bn4,name='Relu3')
        #128x128x3
        convT5=tf.layers.conv2d_transpose(relu4, filters=3, kernel_size=5, strides=(2, 2), padding='valid',name='ConvT5')
        BottleNeck=tf.nn.tanh(convT5,name='BottleNeck')
        output=tf.multiply(BottleNeck,256,name='RGB_out')

Gen_out=generator()
sess=tf.Session()
train_writer = tf.summary.FileWriter('summary',sess.graph)
