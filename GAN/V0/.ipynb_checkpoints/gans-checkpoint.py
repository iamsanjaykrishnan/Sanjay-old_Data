import numpy as np
import tensorflow as tf
from PIL import Image

cakes_np=np.load('E:/Tensorflow/GAN/V0/nparray_cakes.npy')

def cakes_batch(bsize):
    a=np.random.randint(0, high=99, size=bsize)
    cakes_batch_np=[]
    i=0
    for x in a:
        cakes_batch_np.append(cakes_np[x])
    cakes_batch_np=np.array(cakes_batch_np)
    return(cakes_batch_np)


# create tensorflow generator model
def generator(inp1):
    # input varies from 0 to 1 annd the output too
    s1,s2,s3,s4=4,8,64,128
    c1,c2,c3,c4=128,64,32,16


    #dense
    with tf.variable_scope('generator'):
        h1=tf.layers.dense(inp1,s1*s1*3,name='denseh1')
        reshape=tf.reshape(h1,[-1,s1,s1,3])
        #4x4x128
        convT1 = tf.layers.conv2d_transpose(reshape, filters=c1, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT1')
        bn1 = tf.layers.batch_normalization(convT1,name='Bnorm1')
        relu1 = tf.nn.relu(bn1, name='Relu1')
        # 8x8x64
        convT2 = tf.layers.conv2d_transpose(relu1, filters=c2, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT2')
        bn2 = tf.layers.batch_normalization(convT2, name='Bnorm2')
        relu2 = tf.nn.relu(bn2, name='Relu2')
        # 32x32x32
        convT3 = tf.layers.conv2d_transpose(relu2, filters=c3, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT3')
        bn3 = tf.layers.batch_normalization(convT3, name='Bnorm3')
        relu3 = tf.nn.relu(bn3, name='Relu3')
        # 64x64x16
        convT4 = tf.layers.conv2d_transpose(relu3, filters=c4, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT4')
        bn4 = tf.layers.batch_normalization(convT4, name='Bnorm4')
        relu4 = tf.nn.relu(bn4, name='Relu4')
        # 128x128x3
        convT5 = tf.layers.conv2d_transpose(relu4, filters=3, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT5')
        BottleNeck = tf.nn.tanh(convT5, name='BottleNeck')
        const_256 = tf.constant(255,dtype=tf.float32, name='const_256')
        output = tf.multiply(BottleNeck, const_256, name='RGB_out')
        return output
def discriminator(inp_disc,reuse=False):
    # input varies from 0 to 1 annd the output too
    c1,c2,c3,c4=64,128,256,512
    with tf.variable_scope('Discriminator')as scope:
        if reuse:
            scope.reuse_variables()
        #4x4x128
        const_256 = tf.constant(255, dtype=tf.float32, name='const_256')
        squash=tf.div(inp_disc,const_256, name='sqash_inp')
        convT1 = tf.layers.conv2d(squash, filters=c1, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT1')
        bn1 = tf.layers.batch_normalization(convT1,name='Bnorm1')
        relu1 = tf.nn.relu(bn1, name='Relu1')
        # 8x8x64
        convT2 = tf.layers.conv2d(relu1, filters=c2, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT2')
        bn2 = tf.layers.batch_normalization(convT2, name='Bnorm2')
        relu2 = tf.nn.relu(bn2, name='Relu2')
        # 32x32x32
        convT3 = tf.layers.conv2d(relu2, filters=c3, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT3')
        bn3 = tf.layers.batch_normalization(convT3, name='Bnorm3')
        relu3 = tf.nn.relu(bn3, name='Relu3')
        # 64x64x16
        convT4 = tf.layers.conv2d(relu3, filters=c4, kernel_size=5, strides=(2, 2), padding='SAME',name='ConvT4')
        bn4 = tf.layers.batch_normalization(convT4, name='Bnorm4')
        discrim = tf.nn.relu(bn4, name='Relu4')

        return discrim

inp1 = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='input_gen')
Fake_img = generator(inp1)
Real_img=tf.placeholder(dtype=tf.float32,shape=[None,128,128,3],name='input_discrimRGB')

D_fake = discriminator(Fake_img)
D_real = discriminator(Real_img,reuse=True)
#print([x.name for x in tf.global_variables()])
#print([x.name for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Discriminator')])
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)
D_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Discriminator')
G_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss,var_list=D_var))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss,var_list=G_var))

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
version = 'newCake'
save_path = saver.save(sess, 'E:/Tensorflow/GAN/V0/saver/model.ckpt')
ckpt = tf.train.latest_checkpoint('E:/Tensorflow/GAN/V0/saver/' + version)
saver.restore(sess, save_path)
batch_size = 5
EPOCH=100000
train_writer = tf.summary.FileWriter('summary',sess.graph)
for i in range(EPOCH):
    train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
    d_iters = 5
    for k in range(d_iters):
        _, dLoss = sess.run([D_solver, D_loss],feed_dict={inp1: train_noise, Real_img: cakes_batch(batch_size)})
    g_iters = 1
    for k in range(g_iters):
        _, gLoss = sess.run([G_solver, G_loss],feed_dict={inp1: train_noise})
    if i % 500 == 0:
        saver.save(sess, 'E:/Tensorflow/GAN/V0/saver/'+ version + '/' + str(i))
    if i % 50 == 0:
        sample_noise = np.random.uniform(-1.0, 1.0, size=[1, 100]).astype(np.float32)
        imgtest = sess.run(Fake_img, feed_dict={inp1: sample_noise})
        imgtest=np.array(imgtest)
        outfile = 'nparray_cakes'+ str(i)
        np.save(outfile, imgtest)
        im = Image.fromarray(imgtest[0].astype('uint8'))
        im.save(outfile+".jpeg")
# save images
