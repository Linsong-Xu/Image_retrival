import os
import sys
import argparse
import pickle
from utils import *

parser = argparse.ArgumentParser(prog='get_siftdes_list.py', description='get the siftdes of all images')
parser.add_argument('--train', dest='train_path', help='the path of trainning data')
args = parser.parse_args(sys.argv[1:])

train_path = args.train_path
pics = os.listdir(train_path)

parameter_path = './pickle_data/'

if not os.path.isdir(parameter_path):
	os.makedirs(parameter_path)

if __name__ == '__main__':
	images = get_images_path(train_path, pics)
	des_list = get_all_sift_des(images)
	file = parameter_path + 'des_list.pickle'
	with open(file, 'wb') as f:
		pickle.dump(des_list, f)
	print('get all SIFT descriptors')
	
