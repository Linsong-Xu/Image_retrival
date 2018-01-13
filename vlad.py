import os
import sys
import cv2
import argparse
import pylab as pl
import pickle
from utils import *

parser = argparse.ArgumentParser(prog='vlad.py', description='use VLAD for image retrival')
parser.add_argument('--train', dest='train_path', help='the training image path')
parser.add_argument('--query', dest='test_path', help='the query image')
args = parser.parse_args(sys.argv[1:])

train_path = args.train_path
query_img = args.test_path
pics = os.listdir(train_path)


if __name__ == '__main__':

	with open('./pickle_data/SIFTdes.pickle', 'rb') as f:
		SIFTdes = pickle.load(f)
	print('load SIFTdes')
	with open('./pickle_data/vlad_idx.pickle', 'rb') as f:
		vlad_idx = pickle.load(f)
	print('load vlad_idx')
	with open('./pickle_data/tree.pickle', 'rb') as f:
		tree = pickle.load(f)
	print('load tree')

	img = [query_img]
	vlad, img_path = get_VLAD_descriptors(SIFTdes, img)

	dist, idx = tree.query(vlad, 5)

	print('query top-5 finished')
	#display
	img = cv2.imread(query_img)
	pl.figure()
	pl.gray()
	pl.subplot(1,6,1)
	pl.imshow(img)
	pl.axis('off')
	for i in range(idx.shape[1]):
		retrival = cv2.imread(vlad_idx[idx[0,i]])
		pl.gray()
		pl.subplot(1,6,i+2)
		pl.imshow(retrival)
		pl.axis('off')
	pl.show()
