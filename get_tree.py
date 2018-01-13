
import os
import sys
import argparse
import pickle
from utils import *

parser = argparse.ArgumentParser(prog='get_tree.py', description='use balltree to retrival')

parameter_path = './pickle_data/'

if not os.path.isdir(parameter_path):
	os.makedirs(parameter_path)

if __name__ == '__main__':

	with open('./pickle_data/vlad.pickle', 'rb') as f:
		vlad = pickle.load(f)

	tree = balltree(vlad, 40)
	file = parameter_path + 'tree.pickle'
	with open(file, 'wb') as f:
		pickle.dump(tree, f)
	print('balltree finished')
