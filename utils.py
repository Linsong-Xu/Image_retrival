import numpy as np
import cv2
import os
import itertools
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.neighbors import BallTree


def get_images_path(train_path, pics):
	images = []
	for name in pics:
		images.append(os.path.join(train_path,name))
	return images

def get_sift(img_path):
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)
	return kp, des

def get_all_sift_des(images):
	des_list = []
	for i in range(len(images)):
		if (i+1)%600 == 0:
			print('SIFTed:{}'.format((i+1)/600))
		kp, des = get_sift(images[i])
		if des is not None:
			des_list.append(des)
	des_list = list(itertools.chain.from_iterable(des_list))
	des_list = np.array(des_list)
	return des_list

'''
def get_VLAD_descriptors(SIFTdes, images):
	descriptors = []
	img_path = []
	for i in range(len(images)):
		if i%600 == 0:
			print('VLAD:{}'.format(i/600))
		kp, des = get_sift(images[i])
		if des is not None:
			nearest = SIFTdes.predict(des)
			centers = SIFTdes.cluster_centers_
			k = SIFTdes.n_clusters

			m, d = des.shape
			vlad = np.zeros((k,d))

			for j in range(k):
				if np.sum(nearest == j) > 0:
					vlad[j] = np.sum(des[nearest==j, :] - centers[j], axis=0)
			vlad = vlad.flatten()
			vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
			vlad = vlad/np.sqrt(np.dot(vlad, vlad))

			descriptors.append(vlad)
			img_path.append(images[i])
	return descriptors, img_path
'''

def get_VLAD_descriptors(SIFTdes, images):
	descriptors = []
	img_path = []
	for i in range(len(images)):
		if (i+1)%600 == 0:
			print('VLAD:{}'.format((i+1)/600))
		kp, des = get_sift(images[i])
		if des is not None:

			nearest = vq(des, SIFTdes)
			centers = SIFTdes
			k = SIFTdes.shape[0]

			m, d = des.shape
			vlad = np.zeros((k,d))

			for j in range(k):
				if np.sum(nearest[0] == j) > 0:
					vlad[j] = np.sum(des[nearest[0]==j, :] - centers[j], axis=0)
			vlad = vlad.flatten()
			vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
			vlad = vlad/np.sqrt(np.dot(vlad, vlad))

			descriptors.append(vlad)
			img_path.append(images[i])
	return descriptors, img_path

'''
def use_cluster(descriptors, k):
	kmeans = KMeans(n_clusters = k, init = 'k-means++', tol = 0.0001, verbose = 1).fit(descriptors)
	return kmeans
'''

def use_cluster(descriptors, k):
	whitened = whiten(descriptors)
	codebook, variance = kmeans(whitened, k, 1)
	print('kmeans finished and variance:{}'.format(variance))
	return codebook

def balltree(vlad, leaf=40):
	tree = BallTree(vlad, leaf_size = leaf)
	return tree
