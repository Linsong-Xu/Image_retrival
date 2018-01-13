import os
import sys
import argparse
import pickle
from utils import *

parser = argparse.ArgumentParser(prog='train_vlad.py', description='use VLAD for image retrival,train some neccessary data')
parser.add_argument('--train', dest='train_path', help='the path of trainning data')
args = parser.parse_args(sys.argv[1:])

train_path = args.train_path
pics = os.listdir(train_path)

parameter_path = './pickle_data/'

if not os.path.isdir(parameter_path):
	os.makedirs(parameter_path)

if __name__ == '__main__':

	with open('./pickle_data/SIFTdes.pickle', 'rb') as f:
		SIFTdes = pickle.load(f)
	print('have load SIFTdes')

	images = get_images_path(train_path, pics)
	vlad, each_img = get_VLAD_descriptors(SIFTdes, images[0:1000])

	file = parameter_path + 'vlad.pickle'
	with open(file, 'wb') as f:
		pickle.dump(vlad, f)
	print('saved vlad')

	file = parameter_path + 'vlad_idx.pickle'
	with open(file, 'wb') as f:
		pickle.dump(each_img, f)
	print('saved vlad_idx')
