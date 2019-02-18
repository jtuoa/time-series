import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from algorithms import *
import os
import argparse
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import pdb
import pickle

parser = argparse.ArgumentParser(description="main.py")
parser.add_argument("--path", type=str, default="", help="learner method")
#parser.add_argument("--learn", type=str, default="", help="learner method")
args = parser.parse_args()

np.random.seed(42) #set seed for reproducibility

def has_infection(ytrain, ytest):
	yes = np.sum(ytrain) + np.sum(ytest)
	no = ytrain.shape[0] + ytest.shape[0] - yes
	return yes, no

def visualize_sparsity(pid, data):
	#create sub	arr of pid only
	data_arr = np.array(data)
	match = [x for x in list(data_arr) if x[0] == pid]
	match_arr = np.array(match)
	x_time = match_arr[:,3]/1440 #convert min to day
	y_test = match_arr[:,1]
	return x_time, y_test			

'''def gen_kf(kf_dict):
	for n in range(len(kf_dict.keys())):
		yield kf_dict[n]
		
def skfsplit(x, y, kf_dictSelect):
	K = 10
	kf = StratifiedKFold(n_splits=K)
	#kf = KFold(n_splits=K)
	gen = kf.split(x, y)
	y = to_categorical(y)
	y = y.argmax(1)
	if kf_dictSelect:
		#load to check kf contents
		infile = open('kfsplits_%d'%K, 'rb') #open file for read
		new_dict = pickle.load(infile) #load file into new_dict
		infile.close()
		gen = gen_kf(new_dict)
	else:
		kf_dict = {}		
		for split, (train_index, test_index) in enumerate(kf.split(x, y)):
			kf_dict[split] = (train_index, test_index)
		pdb.set_trace()
		outfile = open('kfsplits_%d'%K, 'wb') #open file for write(on disk)
		pickle.dump(kf_dict, outfile) #write kf_dict to outfile
		outfile.close()
	
	splits = {}
	for split, (train_index, test_index) in enumerate(gen):
		splits[split] = (train_index, test_index)
	
	return splits'''

def main():
	#preprocess: data
	'''pdb.set_trace()
	data = data_preprocess("WoundInf_Train_Tests.tsv", "WoundInf_Train_Labels.tsv", ntopTests=20, nYob=12, ntestRel_useRange=True)
	data.load_data()
	data.keep_ntopTests() #NOTE: test names % included
	data.combine_datasets()
	data.convert_infection2()
	data.create_surgery_test_dateCol()
	data.create_surgery_infection_dateCol()
	data.discard_testRelRange()
	data.encode_gender()
	data.integer_encode_testName()
    #['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
	xy, rel2, testName_dict = data.result()
	print("xy shape: ", xy.shape)'''
    
	#plot: visualize data sparsity
	'''pdb.set_trace()
	pid = xy.iloc[0][0] #pick any
	x_time, y_test = visualize_sparsity(pid, xy)
	plt.scatter(x_time, y_test, color='b')
	plt.xlabel('Day')
	plt.ylabel('Test Encode')
	plt.grid()'''

	
	#impute: preprocess
	'''pdb.set_trace()
	impute = impute_preprocess(xy, rel2, nSamples=10, method='NN')
	data_impute = impute.impute_data()
	pdb.set_trace()
	np.save('NN_20tests_10samples.npy', data_impute) #save file	'''	
	
	#prepare: data for MLP
	'''pdb.set_trace()
	#(856, 50, 60, 7) [PID, test_type, nSamples, col]
	data = np.load(os.path.join(args.path,'LOCF_20tests_10samples.npy'))
	
	data_MLP = prepareData_MLP(data)
	#Xtrain, ytrain, Xtest, ytest = data_MLP.xy_simple()
	Xtrain, ytrain, Xtest, ytest = data_MLP.conv_xy_simple()
	np.save('convLOCF_20tests_10samples_Xtrain_simple.npy', Xtrain)
	np.save('convLOCF_20tests_10samples_ytrain_simple.npy', ytrain)
	np.save('convLOCF_20tests_10samples_Xtest_simple.npy', Xtest)
	np.save('convLOCF_20tests_10samples_ytest_simple.npy', ytest)'''
	
	pdb.set_trace()
	Xtrain = np.load(os.path.join(args.path,'convLOCF_20tests_10samples_Xtrain_simple.npy'))
	ytrain = np.load(os.path.join(args.path,'convLOCF_20tests_10samples_ytrain_simple.npy'))
	Xtest = np.load(os.path.join(args.path,'convLOCF_20tests_10samples_Xtest_simple.npy'))
	ytest = np.load(os.path.join(args.path,'convLOCF_20tests_10samples_ytest_simple.npy'))
	
	#imbalance data: yes = 183, no = 673
	yes_infection, no_infection = has_infection(ytrain, ytest)
	print("Yes, No infection", yes_infection, no_infection)
	
	
	pdb.set_trace()
	#Select baseline algorithm 
	#algs = MLP(Xtrain, ytrain)
	#algs = LR()
	algs = CNN(Xtrain)
	
	
	#stratify kfold
	'''K = 10
	#splits = skfsplit(Xtrain, ytrain, 0) #1 = gon't gen kfsplit	
	skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)	
	count = 0
	tpr = []
	tnr = []
	kappa = []
	auc = []
	ytrain = to_categorical(ytrain)
	ytrain = ytrain.argmax(1)
		
	for train_indices, val_indices in skf.split(Xtrain, ytrain):
		print("Training on fold", count)
		print("TRAIN:", train_indices, "VAL:", val_indices)		
		#Generate batches from indices
		x_train, x_val = Xtrain[train_indices], Xtrain[val_indices]
		y_train, y_val = ytrain[train_indices], ytrain[val_indices]
		
		y_train = to_categorical(y_train)
		y_val = to_categorical(y_val)
		algs.train(x_train, y_train, x_val, y_val)
		#pdb.set_trace()
		
		ypred = algs.predict(Xtest)
		ytest = to_categorical(ytest)
		ytest = ytest.argmax(1)
		cm= confusion_matrix(ytest, ypred)
		curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))
		tpr.append(curr_tpr)
		curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1]))
		tnr.append(curr_tnr)

		curr_kappa = metrics.cohen_kappa_score(ytest, ypred)
		kappa.append(curr_kappa)
		
		precision, recall, _ = metrics.precision_recall_curve(ytest, ypred)
		curr_auc = metrics.auc(recall, precision)
		auc.append(curr_auc)
		
		print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f" % (curr_tpr, curr_tnr, curr_kappa, curr_auc))
		#pdb.set_trace()
		count += 1
	
	pdb.set_trace()
	tpr_kFold = sum(tpr)/K
	tnr_kFold = sum(tnr)/K
	kappa_kFold = sum(kappa)/K
	auc_kFold = sum(auc)/K
	print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f" % (tpr_kFold, tnr_kFold, kappa_kFold, auc_kFold))
	pdb.set_trace()'''
	

	#non-stratify kfold
	ytest = to_categorical(ytest)
	ytrain = to_categorical(ytrain)
	ytrain = ytrain.argmax(1) #for LR
	history = algs.train(Xtrain, ytrain)	
	#Predict test set
	ypred = algs.predict(Xtest)
	ytest = ytest.argmax(1)
	#confusion matrix
	cm = confusion_matrix(ytest, ypred)
	print(cm)	
	#precision-recall
	precision, recall, _ = metrics.precision_recall_curve(ytest, ypred)
	f1 = metrics.f1_score(ytest, ypred) 
	auc = metrics.auc(recall, precision)
	ap = metrics.average_precision_score(ytest, ypred)
	print("f1=%.3f auc=%.3f ap=%.3f" %(f1, auc, ap))	
	kappa = metrics.cohen_kappa_score(ytest, ypred)
	print("kappa", kappa)
	
	#Plot: epoch vs. accuracy
	'''plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(['train', 'test'], loc='upper left')
	plt.grid()
	plt.savefig('NNmodel_acc')
	plt.close()
	
	#Plot: epoch vs. loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title("model loss")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.legend(['train', 'test'], loc='upper left')
	plt.grid()
	plt.savefig('NNmodel_loss')
	plt.close()'''
	
	#Plot: ROC curve
	fpr, tpr, th = metrics.roc_curve(ytest, ypred)
	roc_auc = metrics.roc_auc_score(ytest, ypred)
	plt.plot(fpr, tpr)
	plt.legend(['AUC = %0.2f' % roc_auc], loc='upper left')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.grid()
	plt.savefig('LR_NNmodel_roc')
	plt.close()
	
	#Plot: PR curve
	precision, recall, th = metrics.precision_recall_curve(ytest, ypred)
	pr_auc = metrics.auc(recall, precision)
	plt.plot(recall, precision)
	plt.legend(['AUC = %0.2f' % pr_auc], loc='upper left')
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.grid()
	plt.savefig('LR_NNmodel_pr')
	plt.close()
	
	#external kFold split
	'''for split, (_,(train_index, test_index)) in enumerate(splits.items()):
		x_train, y_train = Xtrain[train_index,...], ytrain[train_index]
		x_val, y_val = Xtrain[test_index,...], ytrain[test_index]
		y_val = to_categorical(y_val)
		y_train = to_categorical(y_train)
		algs.train(x_train, y_train, x_val, y_val)
		
		ypred = algs.predict(x_val)
		y_val = y_val.argmax(1)
		cm= confusion_matrix(y_val, ypred)
		curr_tpr = float(cm[1][1]/(cm[1][1] + cm[0][1]))
		tpr.append(curr_tpr)
		curr_tnr = float(cm[0][0]/(cm[0][0] + cm[1][0]))
		tnr.append(curr_tnr)
		
		curr_kappa = metrics.cohen_kappa_score(y_val, ypred)
		kappa.append(curr_kappa)
		
		precision, recall, _ = metrics.precision_recall_curve(y_val, ypred)
		curr_auc = metrics.auc(recall, precision)
		auc.append(curr_auc)
		
		print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f" % (curr_tpr, curr_tnr, curr_kappa, curr_auc))'''
			
	#pdb.set_trace()
	'''x = np.vstack([Xtrain, Xtest])
	y = np.vstack([ytrain, ytest])
	
	splits = kfsplit(x, 1) #1 = don't generate kfsplit
		
	accuracies = []
	algs = MLP(x)
	
	for split, (_,(train_index, test_index)) in enumerate(splits.items()):
		trainx, trainy = x[train_index,...], y[train_index] 
		testx, testy = x[test_index,...], y[test_index]
		testy = to_categorical(testy)
		trainy = to_categorical(trainy)
		algs.train(trainx, trainy)
		ypredict = algs.predict(testx)
		print("ypredict", ypredict)
		#pdb.set_trace()
		
		testy = testy.argmax(1)
		cur_accuracy = float(sum(ypredict==testy))/testy.shape[0]
		accuracies.append(cur_accuracy)'''

	#snapshot preprocess
	'''s = snapShot(xy)
	s.snapshot_data()
	s.snapshot_plot()'''
	

main()
