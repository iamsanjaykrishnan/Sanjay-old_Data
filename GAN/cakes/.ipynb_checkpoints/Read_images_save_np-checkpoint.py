import os, os.path
import numpy as np
import cv2 as cv
## Read images from a path
path='E:/Tensorflow/GAN/cakes/cake/'
image=[]
i=0
for f in os.listdir(path):
    print(f)
    read_img=cv.imread(path+f)
    read_img=cv.resize(read_img,dsize=(128,128), interpolation = cv.INTER_CUBIC)
    read_img=np.asarray(read_img)
    image.append(read_img)
#plt.imshow(read_img)
image_np=np.array(image)
outfile='nparray_cakes'
np.save(outfile, image_np)