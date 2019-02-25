import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import pdb
from utils import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score, recall_score

#memory calc:
#3000*128*64*64*2(weight/bias)*4(each) /1024/1024/1024 = 11.72 G
#256*64*32, 512*64*16, 512*32*32

#GRU-D reference
#https://github.com/zhiyongc/GRU-D/blob/master/main.py
#https://github.com/Han-JD/GRU-D

def sensitivity(y_true, y_pred): #TP/P 1=good
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred): #TN/N (recall) 1=good
	true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
	possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
	return true_negatives / (possible_negatives + K.epsilon())

def precision(y_true, y_pred): #TP/TP+FP 1=good
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred,0, 1)))
	return true_positives / (predicted_positives + K.epsilon())

def kappa(y_true, y_pred):
	#OA=TN+TP/all
	#EA=((TN+FN)*(TN+FP)/all)+((TP+FN)*(TP+FP)/all)/all
	#ka=(OA-EA)/(1-EA)
	true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	false_negatives = K.sum(K.round(K.clip((y_true) * (1-y_pred), 0, 1)))
	false_positives = K.sum(K.round(K.clip((1-y_true) * (y_pred), 0, 1)))
	_all = true_negatives + true_positives + false_negatives + false_positives
	observe_acc = (true_negatives + true_positives)/_all
	exp_acc1 = ((true_negatives+false_negatives)*(true_negatives+false_positives)/_all)
	exp_acc2 = ((true_positives+false_negatives)*(true_positives+false_positives)/_all)
	exp_acc = (exp_acc1 + exp_acc2)/_all
	return (observe_acc - exp_acc)/(1-exp_acc)
	

class GRUD:
	"""
	GRU-D implement
	"""
	def __init__(self, Xtrain, ytrain):
		self.params={'nbatch':32, 'nepoch':40}
		
		from tensorflow import set_random_seed
		np.random.seed(42)
		set_random_seed(2)
		
		self.model = Sequential()
		self.model.add(GRU(33, return_sequences=True, input_shape=(20,1)))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(2, activation = 'softmax'))
		
		opt = Adam(lr=0.001)
		#dont use accuracy metric
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', sensitivity, specificity, precision, kappa])
		
		
	def train(self, xtrain, ytrain):
		history = self.model.fit(xtrain, ytrain, epochs=self.params['nepoch'], batch_size=self.params['nbatch'], validation_split = 0.2, class_weight=self.class_weights, verbose=2)		
		#history = self.model.fit(xtrain, ytrain, epochs=self.params['nepoch'], batch_size=self.params['nbatch'], validation_data = (xval, yval), class_weight=self.class_weights, verbose=2)		
		return history
	
	def predict(self, xtest):
		ypred = self.model.predict(xtest)
		ypred = ypred.argmax(1)
		return ypred


class MLP:
	"""
	Multilayer perceptron, imputation based
	"""
	def __init__(self, Xtrain, ytrain):
		self.params={'nbatch':32, 'nepoch':40}
		
		#make classifier aware of imbalance data by incorp weight in cost fxn
		#self.class_weights = {0:0.4,1:0.6} #sum to 1 else change reg param
		self.class_weights = {0:1, 1:4.5} #class 1 3.65 times weight of class 0
		#yints = ytrain.reshape(ytrain.shape[0],)
		#self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(yints), y=yints)
		#self.class_weight_dict = dict(enumerate(class_weights))
		
		#To generate the same weights initialization for all methods, keras uses numpy and tf as backend so we need to set tf's seed as well
		from tensorflow import set_random_seed
		np.random.seed(42)
		set_random_seed(2)
		
		self.model = Sequential()
		self.model.add(Dense(256, activation='relu', input_dim=Xtrain.shape[1]))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(512, activation='relu'))
		self.model.add(Dropout(0.5))
		#self.model.add(Dense(512, activation='relu'))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(2, activation='softmax'))
		
		opt = Adam(lr=0.001)
		#dont use accuracy metric
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', sensitivity, specificity, precision, kappa])
		
		
	def train(self, xtrain, ytrain):		
		history = self.model.fit(xtrain, ytrain, epochs=self.params['nepoch'], batch_size=self.params['nbatch'], validation_split = 0.2, class_weight=self.class_weights, verbose=2)	
		#history = self.model.fit(xtrain, ytrain, epochs=self.params['nepoch'], batch_size=self.params['nbatch'], validation_data = (xval, yval), class_weight=self.class_weights, verbose=2)
		#self.model.save("/home/juiwen/Documents/CMPUT659/LOCF_classweight_model.h5")
		return history

	def predict(self, xtest):
		ypred = self.model.predict(xtest)
		ypred = ypred.argmax(1)
		return ypred

class Random:
	"""
	success rate when simply guessing respecting training class distribution
	"""
	def __init__(self):
		self.model = DummyClassifier(strategy = "stratified", random_state=0)
	
	def train(self, xtrain, ytrain):
		ytrain = ytrain.argmax(1)	
		self.model.fit(xtrain, ytrain)
	
	def predict(self, xtest):
		ypred = self.model.predict(xtest)
		#ypred = ypred.argmax(1)
		return ypred
		
class LR:
	"""
	Logistic regression, imputation based
	"""
	def __init__(self):
		self.model = LogisticRegression(random_state=0)
		
	def train(self, xtrain, ytrain):
		ytrain = ytrain.argmax(1)
		self.model.fit(xtrain, ytrain)		
		#self.model.save("/home/juiwen/Documents/CMPUT659/mp1.h5")
		#print(self.model.summary())	
	
	def predict(self, xtest):
		ypred = self.model.predict(xtest)
		#ypred = ypred.argmax(1)
		return ypred


class snapShot:
	"""
	pre-process for snap shot algorithm
	"""
	def __init__(self, x):
		self.x = x
		
	#one snapshot data prep
	def snapshot_data(self):
		self.x = np.array(self.x)
		t = check_unique(self.x[:,3])
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
		
		#get blood test
		self.all_test = []
		for r1, elem1 in enumerate(t_uTest):
			match = [x for x in list(self.x) if x[3] == elem1]
			match_arr = np.array(sorted(match, key=itemgetter(0), reverse=False))
			match_arr_save = np.concatenate((match_arr[:,0].reshape(match_arr.shape[0],1), match_arr[:,3].reshape(match_arr.shape[0],1), match_arr[:,4].reshape(match_arr.shape[0],1), match_arr[:,1].reshape(match_arr.shape[0],1), match_arr[:,11].reshape(match_arr.shape[0],1)), axis=1)
			#save each test in high dim
			self.all_test.append(match_arr_save)

	#one snapshot data prep	 
	def snapshot_plot(self):
		#color code search: plotting different colors in matplotlib 
		c = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 
		m = ['*', 'o', 'v', '^', '<', '1', 's', 'x', 'd']
		for i in range(0, len(self.all_test)):
			cidx = 0
			midx = 0
			#print("Table of unique PID per test")
			#cu = check_unique(xy_3D[i][:,0])
			#print(cu)
			#print("Infection :", list(xy_3D[0][:,0]).index(cu[0]))
			const = abs(min(self.all_test[0][:,3])) + 1e-6
			for j in range(0, len(self.all_test[0])-1):
				print(j)
				#plot same color for same PID
				if (self.all_test[i][j][0] - self.all_test[i][j+1][0]) == 0:
					#x = time, y = number
					plt.plot(((self.all_test[i][j][3])/max(self.all_test[0][:,3])), self.all_test[i][j][2], color=c[cidx], marker=m[midx])
					plt.savefig('myfig'+str(i))
				else:
					pdb.set_trace()
					plt.plot(((self.all_test[i][j][3])/max(self.all_test[0][:,3])), self.all_test[i][j][2], color=c[cidx], marker=m[midx])
					plt.savefig('myfig'+str(i))
					if midx == len(m)-1:
						cidx += 1
						midx = 0
					midx += 1
			
			#for last point
			if self.all_test[i][len(self.all_test[0])-1][0] == self.all_test[i][len(self.all_test[0])-2][0]:
				plt.plot(self.all_test[i][len(self.all_test[0])-1][3], self.all_test[i][len(self.all_test[0])-1][2], color=c[cidx], marker=m[midx])
				plt.savefig('myfig'+str(i))
			else:
				if midx == len(m)-1:
					cidx += 1
					midx = 0
				midx += 1
				plt.plot(self.all_test[i][len(self.all_test[0])-1][3], self.all_test[i][len(self.all_test[0])-1][2], color=c[cidx], marker=m[midx])
				plt.savefig('myfig'+str(i))		

