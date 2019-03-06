import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from utils_cleanData import *
from utils_balanceClass import *
from utils_impute import *
from utils_freqBtest import *
from utils_prepareGRU import *
from utils_prepareMLP import *
from algorithms import *
import os
import argparse
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
#from imblearn.over_sampling import SMOTE
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
	x_time = (match_arr[:,3]/1440)/60 #convert min to day
	y_test = match_arr[:,1]
	return x_time, y_test			


def main():	
	#PREPROCESS: data
	'''pdb.set_trace()
	data = data_preprocess("WoundInf_Train_Tests.tsv", "WoundInf_Train_Labels.tsv", ntopTests=20, nYob=12, ntestRel_useRange=True, numKeepTests=5) #811=allTests
	data.load_data()
	data.keep_ntopTests() #NOTE: test names % included
	#data.keep_oneTest()
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

    
	#plot: visualize data sparsity
	#To change: ntopTests = 811, discard_testRelRange self.xnew = self.x1new
	'''pdb.set_trace()
	pid = xy.iloc[3000][0] #pick any pid [0][0], [3000][0]
	x_time, y_test = visualize_sparsity(pid, xy)
	plt.scatter(x_time, y_test, color='b')
	plt.xlabel('Day')
	plt.ylabel('Available test index')
	plt.grid()
	plt.savefig('DataSparsity2')
	plt.close()'''
		
	#IMPUTE: preprocess
	'''pdb.set_trace()
	impute = impute_preprocess(xy, rel2, nSamples=10, method='LOCF')
	data_impute = impute.impute_data()
	pdb.set_trace()
	np.save('LOCF_postop60_ALP_10samples.npy', data_impute) #save file'''
	
	#plot: imputation rate
	'''pdb.set_trace()
	impute_PID = np.load(os.path.join(args.path,'LOCF_keep5Test_count_imputePID.npy'))
	PID = range(0, len(impute_PID))
	#plt.plot(PID, impute_PID)
	plt.scatter(PID, impute_PID)
	plt.xlabel("patient index")
	plt.ylabel("number of imputations")
	plt.grid()
	plt.savefig('num_LOCF_kepp5Test')
	plt.close()'''
	
	#plot: visualize before & after imputation
	'''pdb.set_trace()
	pid = xy.iloc[0][0] #[0]=inf, [2]=no_inf
	testNum = 0
	arr_xy = np.array(xy)
	match = [x for x in list(arr_xy) if x[0] == pid] #get pid
	match_test = [x for x in list(match) if x[1] == testNum]
	match_test = np.array(sorted(match_test, key=itemgetter(3), reverse=True))
	before_imputeX = match_test[:,3]/1440 #bTest time
	before_imputeY = match_test[:,2] #bTest val
	
	after_impute = np.load(os.path.join(args.path,'MEAN_20tests_10samples.npy'))
	pid_idx = np.where(after_impute[:,0] == pid)[0][0]
	matchA = after_impute[pid_idx]
	match_testA = matchA[testNum] #already time sorted
	after_imputeX = match_testA[:,3]/1440
	after_imputeY = match_testA[:,2]
	
	plt.scatter(before_imputeX, before_imputeY, color='blue')
	plt.plot(before_imputeX, before_imputeY, 'b-')
	plt.scatter(after_imputeX, after_imputeY, color='red')
	plt.plot(after_imputeX, after_imputeY, color='red')
	
	pdb.set_trace()
	#second pid
	pid = xy.iloc[2][0]
	arr_xy = np.array(xy)
	match = [x for x in list(arr_xy) if x[0] == pid] #get pid
	match_test = [x for x in list(match) if x[1] == testNum]
	match_test = np.array(sorted(match_test, key=itemgetter(3), reverse=True))
	before_imputeX = match_test[:,3]/1440 #bTest time
	before_imputeY = match_test[:,2] #bTest val
	
	after_impute = np.load(os.path.join(args.path,'MEAN_20tests_10samples.npy'))
	pid_idx = np.where(after_impute[:,0] == pid)[0][0]
	matchA = after_impute[pid_idx]
	match_testA = matchA[testNum] #already time sorted
	after_imputeX = match_testA[:,3]/1440
	after_imputeY = match_testA[:,2]
	
	plt.scatter(before_imputeX, before_imputeY, color='black')
	plt.plot(before_imputeX, before_imputeY, 'k-')
	plt.scatter(after_imputeX, after_imputeY, color='green')
	plt.plot(after_imputeX, after_imputeY, color='green')
	
	#plt.title("before_after imputation")
	plt.ylabel("Blood test results")
	plt.xlabel("Days")
	plt.legend(['Before impute not SSI', 'After impute not SSI', 'Before impute SSI', 'After impute SSI'], loc='best')
	plt.grid()
	plt.savefig('MEAN_before_after_impute')
	plt.close()'''
		
	#CREATE: frequency blood test table (non-impute)
	'''pdb.set_trace()
	freq = frequency_bTest(xy)	
	data_freq = freq.create_freqTable()
	np.save('freqTable_input_20tests.npy', data_freq)'''
	
	#PREPARE: data for GRUD
	'''pdb.set_trace()
	data_GRUD = prepareData_GRUD(xy, ntopTests=20)
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
	data = np.load(os.path.join(args.path,'LOCF_postop60_ALP_10samples.npy'))
	data_MLP = prepareData_MLP(data)
	Xtrain, ytrain, Xtest, ytest = data_MLP.xy_simple()
	#Xtrain, ytrain, Xtest, ytest = data_MLP.conv_xy_simple(test_num=0)
	#Xtrain, ytrain, Xtest, ytest = data_MLP.xy_freq()
	np.save('LOCF_postop60_ALP_10samples_Xtrain_simple.npy', Xtrain)
	np.save('LOCF_postop60_ALP_10samples_ytrain_simple.npy', ytrain)
	np.save('LOCF_postop60_ALP_10samples_Xtest_simple.npy', Xtest)
	np.save('LOCF_postop60_ALP_10samples_ytest_simple.npy', ytest)'''
	
	#LOAD: GRU data
	pdb.set_trace()
	with open('RNN_20tests_10samples_Xtrain.data', 'rb') as fp:
		Xtrain = pickle.load(fp)
	with open('RNN_20tests_10samples_ytrain.data', 'rb') as fp:
		ytrain = pickle.load(fp)
	with open('RNN_20tests_10samples_Xtest.data', 'rb') as fp:
		Xtest = pickle.load(fp)
	with open('RNN_20tests_10samples_ytest.data', 'rb') as fp:
		ytest = pickle.load(fp)
	
	#LOAD: data
	'''pdb.set_trace()
	Xtrain = np.load(os.path.join(args.path,'LOCF_20tests_10samples_Xtrain_simple.npy'))
	ytrain = np.load(os.path.join(args.path,'LOCF_20tests_10samples_ytrain_simple.npy'))
	Xtest = np.load(os.path.join(args.path,'LOCF_20tests_10samples_Xtest_simple.npy'))
	ytest = np.load(os.path.join(args.path,'LOCF_20tests_10samples_ytest_simple.npy'))'''
	
	#ytrain = ytrain.reshape(len(ytrain),1)
	#LOAD: data combine LOCF + Freq
	'''Xtrain2 = np.load(os.path.join(args.path,'LOCF_noDecay_20tests_10samples_Xtrain_simple.npy'))
	ytrain2 = np.load(os.path.join(args.path,'LOCF_noDecay_20tests_10samples_ytrain_simple.npy'))
	Xtest2 = np.load(os.path.join(args.path,'LOCF_noDecay_20tests_10samples_Xtest_simple.npy'))
	ytest2 = np.load(os.path.join(args.path,'LOCF_noDecay_20tests_10samples_ytest_simple.npy'))
	
	Xtrain = np.hstack([Xtrain, Xtrain2])
	Xtest = np.hstack([Xtest, Xtest2])	
	ytrain = ytrain.reshape(len(ytrain),1)
	ytest = ytest.reshape(len(ytest),1)'''
	
	#IMBALANCE: upsample to balance training set
	'''sm = SMOTE(random_state = 42)
	ytrain_sm = to_categorical(ytrain)
	ytrain_sm = train_sm.argmax(1)
	Xtrain_res, ytrain_res = sm.fit_resample(Xtrain, ytrain_sm)
	yes_infection = np.sum(ytrain_sm)
	yes_infection_sm = np.sum(ytrain_res)
	print("Before oversampling", yes_infection, "After oversampling", yes_infection_sm)
	np.save('SMOTE_LOCF_20tests_10samples_Xtrain_simple.npy', Xtrain_res)
	np.save('SMOTE_LOCF_20tests_10samples_ytrain_simple.npy', ytrain_res)'''
	
	#IMBALANCE: downsample to balance training set
	'''pdb.set_trace()
	has_inf, hasnot_inf = has_infection(ytrain, ytest)
	X = np.vstack([Xtrain, Xtest])
	y = np.vstack([ytrain, ytest])

	ds = downSample_preprocess(X, y, has_inf, hasnot_inf)
	Xtrain_hasinf, Xtest_hasinf, x_hasnotinf_sets, Xtest_hasnotinf = ds.train_test_split()
	setIdx = 2
	Xtrain = np.vstack([Xtrain_hasinf, x_hasnotinf_sets[setIdx]])
	Xtest = np.vstack([Xtest_hasinf, Xtest_hasnotinf[0]])
	ytrain = [1]*len(Xtrain_hasinf) + [0]*len(x_hasnotinf_sets[setIdx])
	ytest = [1]*len(Xtest_hasinf) + [0]*len(Xtest_hasnotinf[0])'''

###############################################################################
	
	#Evaluation metric: ttest
	'''avg_acc_DC = np.array([0.7241379310344828, 0.7241379310344828, 0.7011494252873564, 0.7058823529411765, 0.7058823529411765, 0.6588235294117647, 0.6588235294117647, 0.7058823529411765, 0.6352941176470588, 0.7058823529411765])
	avg_acc_LR = np.array([0.8947368421052632, 0.8157894736842105, 0.9473684210526315, 0.9166666666666666, 0.8611111111111112, 0.8611111111111112, 0.8611111111111112, 0.8888888888888888, 0.9166666666666666, 0.8055555555555556])
	avg_acc_MLP = np.array([0.9473684210526315, 1.0, 1.0, 0.9166666666666666, 0.9722222222222222, 1.0, 1.0, 1.0, 1.0, 1.0])
	#tstat, pstat = stats.ttest_rel(avg_acc_DC, avg_acc_LR)
	#tstat, pstat = stats.ttest_rel(avg_acc_LR, avg_acc_MLP)
	import matplotlib.mlab as mlab
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
	ytrain = np.array(ytrain)
	ytest = np.array(ytest)
	yes_infection, no_infection = has_infection(ytrain, ytest)
	print("Yes, No infection", yes_infection, no_infection)
	
	
	#SELECT: algorithm
	pdb.set_trace()
	#algs = Random()
	#algs = LR()
	#algs = MLP(Xtrain, ytrain)
	algs = nonImpute_LSTM(Xtrain, ytrain)
	
	#Xtrain, Xtest kfold, uncomment these for hyperparam
	#Xtrain = np.vstack([Xtrain, Xtest])
	#ytrain = np.vstack([ytrain, ytest])
	#ytrain = np.hstack([ytrain, ytest])
	
	#LSTM: stratify k-fold
	K = 10
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
		#Loop through train_indices
		pdb.set_trace()
		for i in train_indices:
			x_train, y_train = Xtrain[i], ytrain[i]
			x_train = x_train.reshape((1, x_train.shape[0], x_train.shape[1]))
			x_train = np.transpose(x_train, (0,2,1))
			y_train = to_categorical(y_train, num_classes=2)
			y_train = y_train.reshape((1,1,2))
			algs.train(x_train, y_train)
		
		pdb.set_trace()
		for i in val_indices:
			x_val, y_val = Xtrain[i], ytrain[i]
			ypred = algs.predict(x_val)

		pdb.set_trace()
		
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
	pdb.set_trace()
	
	
	#stratify kfold
	'''K = 10
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
		y_val = to_categorical(y_val)
		algs.train(x_train, y_train)
		#pdb.set_trace()
		
		ypred = algs.predict(x_val)
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
	

	#non-kfold
	'''ytest = to_categorical(ytest)
	ytrain = to_categorical(ytrain)
	#ytrain = ytrain.argmax(1) #for LR, Random
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
	pdb.set_trace()
	
	#Plot: epoch vs. accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(['train', 'test'], loc='upper left')
	plt.grid()
	plt.savefig('MLP_LOCFmodel_acc')
	plt.close()
	
	#Plot: epoch vs. loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title("model loss")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.legend(['train', 'test'], loc='upper left')
	plt.grid()
	plt.savefig('MLP_LOCFmodel_loss')
	plt.close()

	#Plot: ROC curve
	fpr, tpr, th = metrics.roc_curve(ytest, ypred)
	roc_auc = metrics.roc_auc_score(ytest, ypred)
	plt.plot(fpr, tpr)
	plt.legend(['AUC = %0.2f' % roc_auc], loc='upper left')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.grid()
	plt.savefig('MLP_LOCFmodel_roc')
	plt.close()
	
	#Plot: PR curve
	precision, recall, th = metrics.precision_recall_curve(ytest, ypred)
	pr_auc = metrics.auc(recall, precision)
	plt.plot(recall, precision)
	plt.legend(['AUC = %0.2f' % pr_auc], loc='upper left')
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.grid()
	plt.savefig('MLP_LOCFmodel_pr')
	plt.close()'''


	#snapshot preprocess
	'''s = snapShot(xy)
	s.snapshot_data()
	s.snapshot_plot()'''
	

main()
