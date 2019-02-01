import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import algorithms as algs
import os
import argparse
from sklearn.model_selection import KFold
import pdb
import pickle

parser = argparse.ArgumentParser(description="main.py")
parser.add_argument("--path", type=str, default="", help="learner method")
#parser.add_argument("--learn", type=str, default="", help="learner method")
args = parser.parse_args()

def gen_kf(kf_dict):
	for n in range(len(kf_dict.keys())):
		yield kf_dict[n]

def kfsplit(x, kf_dictSelect):
	K = 10

	kf = KFold(n_splits=K)
	
	gen = kf.split(x)
	if kf_dictSelect:
		#load to check kf contents
		infile = open('kfsplits_%d'%K, 'rb') #open file for read
		new_dict = pickle.load(infile) #load file into new_dict
		infile.close()
		gen = gen_kf(new_dict)
	else:
		kf_dict = {}		
		for split, (train_index, test_index) in enumerate(kf.split(x)):
			kf_dict[split] = (train_index, test_index)
		pdb.set_trace()
		outfile = open('kfsplits_%d'%K, 'wb') #open file for write(on disk)
		pickle.dump(kf_dict, outfile) #write kf_dict to outfile
		outfile.close()
	
	splits = {}
	for split, (train_index, test_index) in enumerate(gen):
		splits[split] = (train_index, test_index)
	
	return splits

				
	
def main():
	#TODO: preprocess data
	#preprocess data
	'''data = data_preprocess("WoundInf_Train_Tests.tsv", "WoundInf_Train_Labels.tsv", ntopTests=50, nYob=12, ntestRel_useRange=True)
	data.load_data()
	data.keep_ntopTests()
	data.combine_datasets()
	data.delete_infection2()
	data.create_newDataCol()
	data.discard_testRelRange()
	data.create_weightVec_newDataCol()
	data.encode_gender()
	data.create_yobGroup()
	xy, testName = data.result()
	print("xy shape: ", xy.shape)
	np.save('testName.npy', testName)'''
	
	#snapshot preprocess
	'''s = snapShot_preprocess(xy)
	s.snapshot_data()
	s.snapshot_plot()'''
	
	#impute preprocess
	'''impute = impute_preprocess(xy, nSamples = 20)
	data_impute = impute.impute_data()
	np.save('forward_impute_50.npy', data_impute) #save file'''
	
	#finetune preprocess
	'''pdb.set_trace()
	#[pid, time, time_weight, test, numAnswer, gender, yob, infection]
	data_impute = np.load(os.path.join(args.path,'forward_impute_50.npy'))
	testName = np.load(os.path.join(args.path,'testName.npy'))
	data2 = more_preprocess(data_impute, testName)
	data2.get_test_idx()
	data_complete = data2.normalize_test() #(PID, t_uTest, samples, cols)

	pdb.set_trace()
	np.save('forward_impute_complete_50.npy', data_complete) #50 samples'''
	
	#TODO: k-fold split
	pdb.set_trace()
	#(787, 50, 20, 16) [PID, test_type, nSamples, col]
	#col = [PID, date, weight, test_type, result, gender, yob1, yob2, yob3, yob4, yob5, yob6, yob7, yob8, yob9, infection]
	data = np.load(os.path.join(args.path,'forward_impute_complete_50.npy'))
	splits = kfsplit(data, 1) #1 = don't generate kfsplit
	
	#TODO: test algorithm
	accuracies = []
	#How to handle x??
	x = data[:,:,:,0:15] #(787, 50, 20, 15) 
	yd = data[:,:,:,15] #(787,50,20), y[0] = (50,20)
	y = yd[:,0,0] #(787,) per PID
	for split, (_,(train_index, test_index)) in enumerate(splits.items()):
		trainx, trainy = x[train_index,...], y[train_index]
		testx, testy = x[test_index,...], y[test_index]
		algs.train(trainx, trainy, testx, testy)
		ypredict = algs.predict(testx)
		cur_accuracy = float(sum(ypredict==testy))/testy.shape[0]
		accuracies.append(cur_accuracy)	
	
	#TODO: Evaluation
	
	

	pdb.set_trace()


main()
