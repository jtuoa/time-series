import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import argparse
import pdb
import pickle
import scipy.io
from plots import *
from utils_grid import *
from utils_cleanData import *
from utils_balanceClass import *
from utils_impute import *
from utils_freqBtest import *
from utils_prepareGRU import *
from utils_prepareMLP import *
from algorithms import *
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
from sklearn import metrics
from tensorflow.keras.models import load_model
#from imblearn.over_sampling import SMOTE


parser = argparse.ArgumentParser(description="main.py")
parser.add_argument("--path", type=str, default="", help="learner method")
#parser.add_argument("--learn", type=str, default="", help="learner method")
args = parser.parse_args()

np.random.seed(42) #set seed for reproducibility

def has_infection(ytrain, ytest):
	yes = np.sum(ytrain) + np.sum(ytest)
	no = ytrain.shape[0] + ytest.shape[0] - yes
	return yes, no

			
def main():	
	#PREPROCESS: data
	'''pdb.set_trace()
	data = data_preprocess("WoundInf_Train_Tests.tsv", "WoundInf_Train_Labels.tsv", ntopTests=20, nYob=12, ntestRel_useRange=True, numKeepTests=5) #811=allTests
	data.load_data()
	data.keep_ntopTests() #NOTE: test names % included
	data.keep_oneTest()
	data.combine_datasets()
	data.convert_infection2()
	data.create_surgery_test_dateCol()
	data.create_surgery_infection_dateCol()
	data.discard_testRelRange()
	#data.discard_tests_pid()
	data.encode_gender()
	data.integer_encode_testName()
    #['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
	xy, rel2, testName_dict = data.result()
	print("xy shape: ", xy.shape)
	#Top 10 tests: {0: 'ALAT', 1: 'ALP', 2: 'Albumin', 3: 'CRP', 4: 'Hemoglobin', 5: 'Kalium', 6: 'Kreatinin', 7: 'Leukocytter', 8: 'Natrium', 9: 'Trombocytter'}'''
	xy_all = pd.read_pickle('xy_all')
	xy_20 = pd.read_pickle('xy') 
	rel2 = -29519
	
	pdb.set_trace()
	viz = plots_visualize()
	#Fig: data sparsity
	#viz.data_sparsity(xy_all)
	
	#CALC: sampling period
	#fs = grid_preprocess(xy_20)
	#avg_alltest = fs.sample_period_test()
	#count_truemissing, count_lacksampling = fs.classify_missing()
		
	#IMPUTE: preprocess
	'''pdb.set_trace()
	impute = impute_preprocess(xy_20, rel2, nSamples=20, method='LOCF')
	data_impute = impute.impute_data()
	pdb.set_trace()
	np.save('LOCF_20tests_20samples.npy', data_impute) #save file'''
	
	#Fig: imputation rate
	#viz.impute_rate()
	
	#Fig: before after impute
	#viz.before_after_impute(xy_20)
		
	#CREATE: frequency blood test table (non-impute)
	'''pdb.set_trace()
	freq = frequency_bTest(xy_20)	
	data_freq = freq.create_freqTable()
	np.save('freqTable_input_20tests.npy', data_freq)'''
	
	#LOAD: MATLAB data
	#pdb.set_trace()
	#patient_db = scipy.io.loadmat(os.path.join(args.path,'PatientDBFull.mat'))
	
	#PREPARE: data for GRUD
	'''pdb.set_trace()
	has_inf = 183 #per data
	hasnot_inf = 673
	
	data_GRUD = prepareData_GRUD(xy_20, ntopTests=20)
	data_GRUD.upSample(has_inf, hasnot_inf)
	
	Xtrain_GRUDX, ytrain_GRUDX, Xtest_GRUDX, ytest_GRUDX = data_GRUD.create_x()	
	Xtrain_GRUDM, Xtest_GRUDM = data_GRUD.create_mask()
	Xtrain_GRUDD, Xtest_GRUDD = data_GRUD.create_delta()
	Xtrain_GRUD, Xtest_GRUD= data_GRUD.create_input(Xtrain_GRUDX, Xtest_GRUDX, Xtrain_GRUDM, Xtest_GRUDM, Xtrain_GRUDD, Xtest_GRUDD)
	pdb.set_trace()
	with open('RNN_20tests_10samples_Xtrain.data', 'wb') as fp:
		pickle.dump(Xtrain_GRUD, fp)
	with open('RNN_20tests_10samples_ytrain.data', 'wb') as fp:
		pickle.dump(ytrain_GRUDX, fp)
	with open('RNN_20tests_10samples_Xtest.data', 'wb') as fp:
		pickle.dump(Xtest_GRUD, fp)
	with open('RNN_20tests_10samples_ytest.data', 'wb') as fp:
		pickle.dump(ytest_GRUDX, fp)'''

	
	#PREPARE: data for MLP
	'''pdb.set_trace()
	#(856, 20, 10, 7) [PID, test_type, nSamples, col]
	data = np.load(os.path.join(args.path,'LOCF_20tests_20samples.npy'))
	data_MLP = prepareData_MLP(data)
	Xtrain, ytrain, Xtest, ytest = data_MLP.xy_simple()
	#Xtrain, ytrain, Xtest, ytest = data_MLP.conv_xy_simple(test_num=0)
	#Xtrain, ytrain, Xtest, ytest = data_MLP.xy_freq()
	np.save('LOCF_20tests_20samples_Xtrain.npy', Xtrain)
	np.save('LOCF_20tests_20samples_ytrain.npy', ytrain)
	np.save('LOCF_20tests_20samples_Xtest.npy', Xtest)
	np.save('LOCF_20tests_20samples_ytest.npy', ytest)'''
	
	#LOAD: GRU data
	'''pdb.set_trace()
	with open(os.path.join(args.path,'RNN_20tests_10samples_Xtrain.data'), 'rb') as fp:
		Xtrain = pickle.load(fp)
	with open(os.path.join(args.path,'RNN_20tests_10samples_ytrain.data'), 'rb') as fp:
		ytrain = pickle.load(fp)
	with open(os.path.join(args.path,'RNN_20tests_10samples_Xtest.data'), 'rb') as fp:
		Xtest = pickle.load(fp)
	with open(os.path.join(args.path,'RNN_20tests_10samples_ytest.data'), 'rb') as fp:
		ytest = pickle.load(fp)'''
	
	#LOAD: data
	'''pdb.set_trace()
	Xtrain = np.load(os.path.join(args.path,'LOCF_20tests_11samples_Xtrain.npy'))
	ytrain = np.load(os.path.join(args.path,'LOCF_20tests_11samples_ytrain.npy'))
	Xtest = np.load(os.path.join(args.path,'LOCF_20tests_11samples_Xtest.npy'))
	ytest = np.load(os.path.join(args.path,'LOCF_20tests_11samples_ytest.npy'))'''
	
	#LOAD: GPimpute data
	'''Xtrain = pd.read_csv(os.path.join(args.path, 'X_GPimp.csv'))
	Xtrain.fillna(0, inplace=True)
	Xtrain = np.array(Xtrain)
	ytrain = pd.read_csv(os.path.join(args.path, 'Y_GPimp.csv'))
	ytrain.fillna(0, inplace=True)
	ytrain = np.array(ytrain)
	Xtest = pd.read_csv(os.path.join(args.path, 'Xtest_GPimp.csv'))
	Xtest.fillna(0, inplace=True)
	Xtest = np.array(Xtest)
	ytest = pd.read_csv(os.path.join(args.path, 'Ytest_GPimp.csv'))
	ytest.fillna(0, inplace=True)
	ytest = np.array(ytest)'''
	
	#LOAD: GPnonimpute data
	'''Xtrain = pd.read_csv(os.path.join(args.path, 'X_GP.csv'))
	Xtrain = np.array(Xtrain)
	ytrain = pd.read_csv(os.path.join(args.path, 'Y_GP.csv'))
	ytrain = np.array(ytrain)
	Xtest = pd.read_csv(os.path.join(args.path, 'Xtest_GP.csv'))
	Xtest = np.array(Xtest)
	ytest = pd.read_csv(os.path.join(args.path, 'Ytest_GP.csv'))
	ytest = np.array(ytest)'''
	
	#LOAD: Freq data
	pdb.set_trace()
	Xtrain = pd.read_csv(os.path.join(args.path, 'X_freq1.csv'))
	ytrain = pd.read_csv(os.path.join(args.path, 'Y_freq1.csv'))
	Xtest = pd.read_csv(os.path.join(args.path, 'Xtest_freq1.csv'))
	ytest = pd.read_csv(os.path.join(args.path, 'Ytest_freq1.csv'))
		
	#IMBALANCE: Bootstrap upsample to balance by duplicate minority sample + gauss
	'''pdb.set_trace()	
	has_inf = np.sum(ytrain)
	hasnot_inf = len(Xtrain) - has_inf
	us = upSample_preprocess(Xtrain, ytrain, has_inf, hasnot_inf)
	Xtrain, ytrain = us.us_minority(add_noise=1)
	
	has_inf = np.sum(ytest)
	hasnot_inf = len(Xtest) - has_inf
	us = upSample_preprocess(Xtest, ytest, has_inf, hasnot_inf)
	Xtest, ytest = us.us_minority(add_noise=0)'''
	
	#LR Learning Curve
	'''pdb.set_trace()
	Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
	N = 800
	Xtrain_m1 = Xtrain[0:N,:]
	ytrain_m1 = ytrain[0:N]'''
	
	#IMBALANCE: SMOTE upsample to balance training set
	'''sm = SMOTE(random_state = 42)
	ytrain_sm = to_categorical(ytrain)
	ytrain_sm = train_sm.argmax(1)
	Xtrain_res, ytrain_res = sm.fit_resample(Xtrain, ytrain_sm)
	yes_infection = np.sum(ytrain_sm)
	yes_infection_sm = np.sum(ytrain_res)
	print("Before oversampling", yes_infection, "After oversampling", yes_infection_sm)
	np.save('SMOTE_LOCF_20tests_10samples_Xtrain_simple.npy', Xtrain_res)
	np.save('SMOTE_LOCF_20tests_10samples_ytrain_simple.npy', ytrain_res)'''
	
	#IMBALANCE: Downsample to balance training set
	'''pdb.set_trace()
	has_inf, hasnot_inf = has_infection(ytrain, ytest)
	X = np.vstack([Xtrain, Xtest])
	y = np.vstack([ytrain, ytest])

	ds = downSample_preprocess(X, y, has_inf, hasnot_inf)
	Xtrain_hasinf, Xtest_hasinf, Xtrain_hasnotinf_sets, Xtest_hasnotinf = ds.train_test_split(add_noise=1)
	setIdx = 3
	Xtrain = np.vstack([Xtrain_hasinf, Xtrain_hasnotinf_sets[setIdx]])
	ytrain = [1]*len(Xtrain_hasinf) + [0]*len(Xtrain_hasnotinf_sets[setIdx])

	#EVAL Downsample HERE:
	Xtest = np.vstack([Xtest_hasinf, Xtest_hasnotinf[0]])
	ytest = [1]*len(Xtest_hasinf) + [0]*len(Xtest_hasnotinf[0])'''
	'''downSample_LR0 = pickle.load(open(os.path.join(args.path,'MEAN_downsample_LR0.sav'), 'rb'))
	downSample_LR1 = pickle.load(open(os.path.join(args.path,'MEAN_downsample_LR1.sav'), 'rb'))
	downSample_LR2 = pickle.load(open(os.path.join(args.path,'MEAN_downsample_LR2.sav'), 'rb'))
	downSample_LR3 = pickle.load(open(os.path.join(args.path,'MEAN_downsample_LR3.sav'), 'rb'))

	ypred = downSample_LR0.predict(Xtest)
	probs = ypred[:,1] #keep prob for positive outcome only
	fpr, tpr, th = metrics.roc_curve(ytest, probs)
	roc_auc = metrics.roc_auc_score(ytest, probs)
	pdb.set_trace()
	#confusion matrix
	cm = confusion_matrix(ytest, ypred)
	print(cm)
	curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))#TP/TP+FN Sensi
	curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1])) #TN/TN+FP Speci
	curr_acc = float((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1]))
	curr_kappa = metrics.cohen_kappa_score(ytest, ypred)
	precision, recall, _ = metrics.precision_recall_curve(ytest, ypred)
	curr_auc = metrics.auc(recall, precision)
	print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (curr_tpr, curr_tnr, curr_kappa, curr_auc, curr_acc))
	pdb.set_trace()'''
	

###############################################################################
	#Evaluation metric: ttest
	'''pdb.set_trace()
	avg_acc_DC = np.array([0.4117647058823529, 0.4117647058823529, 0.4117647058823529, 0.41044776119402987, 0.41044776119402987, 0.41044776119402987, 0.39552238805970147, 0.41044776119402987, 0.39552238805970147, 0.41044776119402987])
	avg_acc_LR = np.array([0.7941176470588235, 0.8382352941176471, 0.8088235294117647, 0.7761194029850746, 0.7985074626865671, 0.7238805970149254, 0.835820895522388, 0.8134328358208955, 0.8432835820895522, 0.8283582089552238])
	avg_acc_MLP = np.array([0.8308823529411765, 0.8970588235294118, 0.8455882352941176, 0.9626865671641791, 0.9552238805970149, 0.9328358208955224, 0.9477611940298507, 0.9402985074626866, 0.9477611940298507, 0.9626865671641791])
	#tstat, pstat = stats.ttest_rel(avg_acc_DC, avg_acc_LR)
	tstat, pstat = stats.ttest_rel(avg_acc_LR, avg_acc_MLP)'''
	'''import matplotlib.mlab as mlab
	mean = avg_acc_DC.mean()
	sigma = avg_acc_DC.std()
	var = np.var(avg_acc_DC)
	x = np.linspace(-3*sigma+mean, 3*sigma+mean, 100)
	plt.plot(x, mlab.normpdf(x, mean, sigma), color='blue')
	
	mean = avg_acc_LR.mean()
	sigma = avg_acc_LR.std()
	x = np.linspace(-3*sigma+mean, 3*sigma+mean, 100)
	plt.plot(x, mlab.normpdf(x, mean, sigma), color='black')
	
	mean = avg_acc_MLP.mean()
	sigma = avg_acc_MLP.std()
	x = np.linspace(-3*sigma+mean, 3*sigma+mean, 100)
	plt.plot(x, mlab.normpdf(x, mean, sigma), color='red')
	
	plt.grid()
	plt.xlabel("Accuracy")
	plt.legend(['DC var=0.0008', 'LR var=0.0018', 'MLP var=0.00078'], loc='best')
	plt.savefig('bias_var')
	plt.close()
	pdb.set_trace()'''
###############################################################################
	
	#imbalance data: yes = 183, no = 673 / yes = 172 no = 318
	#ytrain = np.array(ytrain)
	#ytest = np.array(ytest)
	#yes_infection, no_infection = has_infection(ytrain, ytest)
	#print("Yes, No infection", yes_infection, no_infection)
	
	
	#SELECT: algorithm
	pdb.set_trace()
	#algs = Random()
	#algs = LR()
	algs = MLP(Xtrain, ytrain)
	#algs = nonImpute_LSTM(Xtrain, ytrain)
	
	#Xtrain, Xtest kfold, comment these for hyperparam
	#Xtrain = np.vstack([Xtrain, Xtest])
	#ytrain = np.vstack([ytrain, ytest])
	#ytrain = np.hstack([ytrain, ytest])
	
	#LSTM: stratify k-fold
	'''K = 2
	#splits = skfsplit(Xtrain, ytrain, 0) #1 = gon't gen kfsplit	
	skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)	
	count = 0
	tpr = []
	tnr = []
	acc = []
	kappa = []
	auc = []

	#flatten the patient matrix 1x 6xx 
	for train_indices, val_indices in skf.split(Xtrain, ytrain):
		print("Training on fold", count)
		print("TRAIN:", train_indices, "VAL:", val_indices)		
		pdb.set_trace()
		for i in train_indices:
			print("iteration", i)
			x_train, y_train = Xtrain[i], ytrain[i] #pid
			x_train = x_train.reshape((1, x_train.shape[0], x_train.shape[1]))
			#x_train = np.transpose(x_train, (0,2,1))
			y_train = to_categorical(y_train, num_classes=2)
			y_train = y_train.reshape((1,2))
			#pdb.set_trace()
			algs.train(x_train, y_train)
		
		pdb.set_trace()
		yp = []
		yv = []
		for i in val_indices:
			x_val, y_val = Xtrain[i], ytrain[i]
			x_val = x_val.reshape((1, x_val.shape[0], x_val.shape[1]))
			#x_val = np.transpose(x_val, (0,2,1))
			ypred = algs.predict(x_val)
			ypred = ypred.argmax(1)
			print("actual, pred", y_val, ypred)
			yp.append(ypred)
			yv.append(y_val)

		pdb.set_trace()	
		yp = np.array(yp)
		yv = np.array(yv)	
		cm= confusion_matrix(yv, yp)
		curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))#TP/TP+FN Sensi
		tpr.append(curr_tpr)
		curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1])) #TN/TN+FP Speci
		tnr.append(curr_tnr)
		curr_acc = float((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1]))
		acc.append(curr_acc)

		curr_kappa = metrics.cohen_kappa_score(yv, yp)
		kappa.append(curr_kappa)
		
		precision, recall, _ = metrics.precision_recall_curve(yv, yp)
		curr_auc = metrics.auc(recall, precision)
		auc.append(curr_auc)
		
		print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (curr_tpr, curr_tnr, curr_kappa, curr_auc, curr_acc))
		#pdb.set_trace()
		count += 1

	pdb.set_trace()
	tpr_kFold = sum(tpr)/K
	tnr_kFold = sum(tnr)/K
	kappa_kFold = sum(kappa)/K
	auc_kFold = sum(auc)/K
	acc_kFold = sum(acc)/K
	print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (tpr_kFold, tnr_kFold, kappa_kFold, auc_kFold, acc_kFold))
	pdb.set_trace()'''
	
	
	#stratify kfold
	'''K = 5
	#splits = skfsplit(Xtrain, ytrain, 0) #1 = gon't gen kfsplit	
	skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)	
	count = 0
	tpr = []
	tnr = []
	acc = []
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
		#y_val = to_categorical(y_val)
		algs.train(x_train, y_train)
		#pdb.set_trace()
		
		ypred = algs.predict(x_val)
		#probs = ypred[:,1] #keep prob for positive outcome only
		#roc_auc = metrics.roc_auc_score(y_val, probs)
		#print("auc = %0.3f" % roc_auc)
		ytest = y_val.argmax(1)
		cm= confusion_matrix(ytest, ypred)
		curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))#TP/TP+FN Sensi
		tpr.append(curr_tpr)
		curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1])) #TN/TN+FP Speci
		tnr.append(curr_tnr)
		curr_acc = float((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1]))
		acc.append(curr_acc)

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
	acc_kFold = sum(acc)/K
	print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (tpr_kFold, tnr_kFold, kappa_kFold, auc_kFold, acc_kFold))
	pdb.set_trace()'''
	
	
	#LSTM: non-kfold
	'''pdb.set_trace()
	#ytrain = to_categorical(ytrain)
	for i in range(0, len(Xtrain)):
		X_train, y_train = Xtrain[i], ytrain[i]
		X_train = X_train.reshape((1, X_train.shape[0], X_train.shape[1]))
		y_train = to_categorical(y_train, num_classes=2)
		y_train = y_train.reshape((1,2))			
		history = algs.train(X_train, y_train)
				
	pdb.set_trace()	
	#Predict test set
	yp = []
	for i in range(0, len(Xtest)):
		X_test = Xtest[i]
		X_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1]))
		ypred = algs.predict(X_test)
		ypred = ypred.argmax(1)
		ypred = ypred[0]
		print("pred", ypred)
		yp.append(ypred)
	pdb.set_trace()
	#AUC
	ypred_arr = np.array(yp[0])
	for i in range(1, len(yp)):
		ypred_arr = np.vstack([ypred_arr, yp[i]])
	probs = ypred_arr[:,1]
	roc_auc = metrics.roc_auc_score(ytest, probs)
	pdb.set_trace()
	#confusion matrix
	cm = confusion_matrix(ytest, yp)
	print(cm)
	curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))#TP/TP+FN Sensi
	curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1])) #TN/TN+FP Speci
	curr_acc = float((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1]))
	print("tpr= %0.3f, tnr= %0.3f, auc= %0.3f, acc= %0.3f" % (curr_tpr, curr_tnr, roc_auc, curr_acc))
	pdb.set_trace()'''
	
	#non-kfold
	pdb.set_trace()
	#algs = load_model(os.path.join(args.path,'LOCF_MLP_model.h5'))
	#algs = pickle.load(open(os.path.join(args.path,'LOCF_learnCurve800_LR4.sav'), 'rb'))
	ytrain = to_categorical(ytrain)
	history = algs.train(Xtrain, ytrain)
	ypred = algs.predict(Xtest)
	probs = ypred[:,1] #keep prob for positive outcome only
	roc_auc = metrics.roc_auc_score(ytest, probs)
	print("auc = %0.3f" % roc_auc)
	pdb.set_trace()
	ytest = to_categorical(ytest)
	ytest = ytest.argmax(1)
	#confusion matrix
	ypred = ypred.argmax(1) #MLP
	cm = confusion_matrix(ytest, ypred)
	print(cm)
	curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))#TP/TP+FN Sensi
	curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1])) #TN/TN+FP Speci
	curr_acc = float((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1]))
	curr_kappa = metrics.cohen_kappa_score(ytest, ypred)
	#precision, recall, _ = metrics.precision_recall_curve(ytest, ypred)
	#curr_auc = metrics.auc(recall, precision)
	fpr, tpr, th = metrics.roc_curve(ytest, ypred)
	curr_auc = metrics.roc_auc_score(ytest, ypred)
	print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (curr_tpr, curr_tnr, curr_kappa, curr_auc, curr_acc))
	pdb.set_trace()
	
	#Plot: ROC curve
	'''pdb.set_trace()
	probLRnn = np.load('LRnn_probs.npy') #NN
	ytestnn = np.load('LRnn_ytest.npy')
	probLRlocf = np.load('LRlocf_probs.npy') #LOCF
	ytestlocf = np.load('LRlocf_ytest.npy')
	probLRgp = np.load('LRgp_probs.npy') #GP
	ytestgp = np.load('LRgp_ytest.npy')
	
	probMLPnn = np.load('MLPnn_probs.npy') #NN
	probMLPlocf = np.load('MLPlocf_probs.npy') #LOCF
	probMLPgp = np.load('MLPgp_probs.npy') #GP
	viz.roc_curveA(ytestnn, probLRnn, ytestlocf, probLRlocf, ytestgp, probLRgp, probMLPnn, probMLPlocf, probMLPgp)'''
	
	'''pdb.set_trace()
	probLSTM = np.load('LSTM_probs.npy') #LSTM
	ytestLSTM = np.load('LSTM_ytest.npy')
	probGP_LR = np.load('GPlr_probs.npy') #GP
	ytestGP_LR = np.load('GPlr_ytest.npy')
	probGP_MLP = np.load('GPmlp_probs.npy') #GP
	ytestGP_MLP = np.load('GPmlp_ytest.npy')
	viz.roc_curveB(ytestLSTM, probLSTM, ytestGP_LR, probGP_LR, ytestGP_MLP, probGP_MLP)'''
	
	#Plot: Cost curve
	'''pdb.set_trace()
	cmLRnn = np.load('LRnn_cm.npy')
	cmLRlocf = np.load('LRlocf_cm.npy')
	cmLRgp = np.load('LRgp_cm.npy')
	cmMLPnn = np.load('MLPnn_cm.npy')
	cmMLPlocf = np.load('MLPlocf_cm.npy')
	cmMLPgp = np.load('MLPgp_cm.npy')
	viz.cost_curveA(cmLRnn, cmLRlocf, cmLRgp, cmMLPnn, cmMLPlocf, cmMLPgp)'''
	
	pdb.set_trace()
	cmLSTM = np.load('LSTM_cm.npy')
	cmGPlr = np.load('GPlr_cm.npy')
	cmGPmlp = np.load('GPmlp_cm.npy')
	viz.cost_curveB(cmLSTM, cmGPlr, cmGPmlp)
	

	#snapshot preprocess
	'''s = snapShot(xy)
	s.snapshot_data()
	s.snapshot_plot()'''
	

main()

#################################################################################
#LSTM impute: stratify k-fold
'''K = 10
#splits = skfsplit(Xtrain, ytrain, 0) #1 = gon't gen kfsplit	
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)	
count = 0
tpr = []
tnr = []
acc = []
kappa = []
auc = []
Xtrain, ytrain = np.array(Xtrain), np.array(ytrain)
Xtest, ytest = np.array(Xtest), np.array(ytest)

#flatten the patient matrix 1x 6xx 
for train_indices, val_indices in skf.split(Xtrain, ytrain):
	print("Training on fold", count)
	print("TRAIN:", train_indices, "VAL:", val_indices)	
	#pdb.set_trace()
	x_train, x_val = Xtrain[train_indices], Xtrain[val_indices]
	y_train, y_val = ytrain[train_indices], ytrain[val_indices]		
	x_train = np.transpose(x_train, (0,2,1))
	y_train = to_categorical(y_train, num_classes=2)
	algs.train(x_train, y_train)
	
	#pdb.set_trace()
	#algs = load_model(os.path.join(args.path,'LSTM_model.h5'))
	x_val = np.transpose(x_val, (0,2,1))
	ypred = algs.predict(x_val)
	ypred = ypred.argmax(1)
	print("actual, pred", y_val, ypred)

	#pdb.set_trace()	
	cm= confusion_matrix(y_val, ypred)
	curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))#TP/TP+FN Sensi
	tpr.append(curr_tpr)
	curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1])) #TN/TN+FP Speci
	tnr.append(curr_tnr)
	curr_acc = float((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1]))
	acc.append(curr_acc)

	curr_kappa = metrics.cohen_kappa_score(y_val, ypred)
	kappa.append(curr_kappa)
	
	precision, recall, _ = metrics.precision_recall_curve(y_val, ypred)
	curr_auc = metrics.auc(recall, precision)
	auc.append(curr_auc)
	
	print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (curr_tpr, curr_tnr, curr_kappa, curr_auc, curr_acc))
	#pdb.set_trace()
	count += 1

pdb.set_trace()
tpr_kFold = sum(tpr)/K
tnr_kFold = sum(tnr)/K
kappa_kFold = sum(kappa)/K
auc_kFold = sum(auc)/K
acc_kFold = sum(acc)/K
print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (tpr_kFold, tnr_kFold, kappa_kFold, auc_kFold, acc_kFold))
pdb.set_trace()'''
	
	
#LSTM: stratify k-fold
'''K = 2
#splits = skfsplit(Xtrain, ytrain, 0) #1 = gon't gen kfsplit	
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)	
count = 0
tpr = []
tnr = []
acc = []
kappa = []
auc = []

for train_indices, val_indices in skf.split(Xtrain, ytrain):
	print("Training on fold", count)
	print("TRAIN:", train_indices, "VAL:", val_indices)		
	pdb.set_trace()
	#create array from list indices
	n_subtrain = 2
	subtrain_len = int(len(train_indices)/n_subtrain)
	for n in range(0, n_subtrain):		
		start_trainidx = n * subtrain_len
		end_trainidx = start_trainidx + subtrain_len
		
		x_train = Xtrain[train_indices[start_trainidx]]
		y_train = ytrain[train_indices[start_trainidx]]
		for i in range(start_trainidx, end_trainidx):
			tmpx = Xtrain[train_indices[i]]
			tmpy = ytrain[train_indices[i]]
			x_train = np.vstack([x_train, tmpx])
			y_train = np.vstack([y_train, tmpy])
			
			
		pdb.set_trace()
		x_train = x_train.reshape((1, x_train.shape[0], x_train.shape[1]))
		#x_train = np.transpose(x_train, (0,2,1))
		y_train = to_categorical(y_train, num_classes=2)
		y_train = y_train.reshape((1,y_train.shape[0], y_train.shape[1]))
		#pdb.set_trace()
		algs.train(x_train, y_train)
	
	
				
	pdb.set_trace()
	x_val = Xtrain[val_indices[0]]
	y_val = ytrain[val_indices[0]]
	for i in range(0, len(val_indices)):
		tmpx = Xtrain[val_indices[i]]
		tmpy = ytrain[val_indices[i]]
		x_val = np.vstack([x_val, tmpx])
		y_val = np.vstack([y_val, tmpy])
	pdb.set_trace()
	yp = []
	yv = []
	for i in val_indices:
		x_val, y_val = Xtrain[i], ytrain[i]
		x_val = x_val.reshape((1, x_val.shape[0], x_val.shape[1]))
		x_val = np.transpose(x_val, (0,2,1))
		ypred = algs.predict(x_val)
		ypred = ypred.argmax(1)
		ypred = ypred[0]
		print("actual, pred", y_val, ypred)
		yp.append(ypred)
		yv.append(y_val)

	pdb.set_trace()	
	yp = np.array(yp)
	yv = np.array(yv)	
	cm= confusion_matrix(yv, yp)
	curr_tpr = float(cm[1][1]/(cm[1][1] + cm[1][0]))#TP/TP+FN Sensi
	tpr.append(curr_tpr)
	curr_tnr = float(cm[0][0]/(cm[0][0] + cm[0][1])) #TN/TN+FP Speci
	tnr.append(curr_tnr)
	curr_acc = float((cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0]+cm[0][0]+cm[1][1]))
	acc.append(curr_acc)

	curr_kappa = metrics.cohen_kappa_score(yv, yp)
	kappa.append(curr_kappa)
	
	precision, recall, _ = metrics.precision_recall_curve(yv, yp)
	curr_auc = metrics.auc(recall, precision)
	auc.append(curr_auc)
	
	print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (curr_tpr, curr_tnr, curr_kappa, curr_auc, curr_acc))
	#pdb.set_trace()
	count += 1

pdb.set_trace()
tpr_kFold = sum(tpr)/K
tnr_kFold = sum(tnr)/K
kappa_kFold = sum(kappa)/K
auc_kFold = sum(auc)/K
acc_kFold = sum(acc)/K
print("tpr= %0.3f, tnr= %0.3f, kappa= %0.3f, auc= %0.3f, acc= %0.3f" % (tpr_kFold, tnr_kFold, kappa_kFold, auc_kFold, acc_kFold))
pdb.set_trace()'''


