import os
import cv2
import sys
import numpy as np
import argparse
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn import preprocessing
import pylab as pl

parser = argparse.ArgumentParser(prog='BoW.py', description='use BoW for image retrival')
parser.add_argument('--train', dest='train_path', help='the training image path')
parser.add_argument('--query', dest='test_path', help='the query image')
args = parser.parse_args(sys.argv[1:])

train_path = args.train_path
query_img = args.test_path
pics = os.listdir(train_path)

images = []
for name in pics:
    images.append(os.path.join(train_path,name))

#print(images)


des_list = []
totalSift = 0
for i in range(len(images)):
    if i%600 == 0:
        print('SIFTed:{}'.format(i/600))

    img = cv2.imread(images[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    des_list.append((i, des))
    if des is not None:
        totalSift += des.shape[0]

print('Training images have been SIFTed and totalSift is {}'.format(totalSift))

#vstack is the bottleneck:copy the 'descriptors' and 'descriptor' first
#and then rewrite the new array
'''
descriptors = des_list[0][1]
for i in range(1,len(images)):
    if i%600 == 0:
        print('vstack:{}'.format(i/600))
    descriptor = des_list[i][1]
    if descriptor is not None:
        descriptors = np.vstack((descriptors, descriptor)) 
print('vstack finish')
'''

descriptors = np.zeros((totalSift,128))
index = 0
for i in range(len(images)):
    if i%600 == 0:
        print('concatenate:{}'.format(i/600))
    descriptor = des_list[i][1]
    if descriptor is not None:
        for j in range(descriptor.shape[0]):
            descriptors[index,:] = descriptor[j]
            index += 1

numfeature = 1000
whitened = whiten(descriptors)
print('whitened')
codebook, variance = kmeans(whitened, numfeature, 1)
print('kmeans finished')

print('{} features -> {} variance'.format(numfeature, variance))

img_vec = np.zeros((len(images), numfeature), "float32")
for i in range(len(images)):
    if des_list[i][1] is not None:
        #tmp = whiten(des_list[i][1])
        code, dis = vq(des_list[i][1],codebook)
        for w in code:
            img_vec[i][w] += 1

df = np.sum( (img_vec > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*10000+1) / (1.0*df + 1)), 'float64')
img_vec = img_vec*idf
img_vec = preprocessing.normalize(img_vec, norm='l2')

print('get img_vec')

img = cv2.imread(query_img)
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kpts, des = sift.detectAndCompute(gray, None)

query_vec = np.zeros((1, numfeature), "float64")
#des = whiten(des)
code, dis = vq(des,codebook)
for w in code:
    query_vec[0][w] += 1

query_vec = query_vec*idf
query_vec = preprocessing.normalize(query_vec, norm='l2')

score = np.dot(query_vec, img_vec.T)
rank_ID = np.argsort(-score)

#img = cv2.imread(query_img)
#rank_ID = [1,2,3,4,5,6]
pl.figure()
pl.gray()
pl.subplot(1,6,1)
pl.imshow(img)
pl.axis('off')


for i in range(5):
    retrival = cv2.imread(images[rank_ID[0,i]])
    pl.gray()
    pl.subplot(1,6,i+2)
    pl.imshow(retrival)
    pl.axis('off')

pl.show()  

