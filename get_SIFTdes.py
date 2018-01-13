import os
import sys
import argparse
import pickle
from utils import *

parser = argparse.ArgumentParser(prog='get_SIFTdes.py', description='get the bag of words of all images')

parameter_path = './pickle_data/'

if not os.path.isdir(parameter_path):
	os.makedirs(parameter_path)

if __name__ == '__main__':
	with open('./pickle_data/des_list.pickle', 'rb') as f:
		des_list = pickle.load(f)
    print('load des_list')
        
	SIFTdes = use_cluster(des_list, 1000)

	file = parameter_path + 'SIFTdes.pickle'
	with open(file, 'wb') as f:
		pickle.dump(SIFTdes, f)
	print('get all SIFT descriptors')
