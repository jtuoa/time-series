import numpy as np
import pandas as pd
from operator import itemgetter
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.mlab as mlab
#from hypertools.hypertools.tools.reduce import reduce as reducer
#from ppca.ppca import PPCA
import random
import pdb
import copy

DAY_MIN = 1440
S_MIN = 60


def check_unique(arr):
	u,c = np.unique(arr, return_counts=True)
	uc_table = np.asarray((u,c)).T
	print(uc_table)
	print("uc_table shape: ", uc_table.shape)
	return uc_table

class prepareData_MLP:
	def __init__(self, data):
		#old x: (856, 50, 60, 7) [PID, test_type, nSamples, col]
    	#['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
		self.data = np.array(data)

	def xy_output(self, x_simple, y_simple, is_3D):
		#pdb.set_trace()
		#stratify: preserves 21% ones and 79% zeros in y_simple
		indices = np.arange(len(y_simple))
		Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = train_test_split(x_simple, y_simple, indices, test_size=0.2, random_state=42, stratify=y_simple)
		
		#pdb.set_trace()
		#np.save('train_indices.npy', train_indices)
		#np.save('test_indices.npy', test_indices)
        
		sc_X = StandardScaler() #standardize data
		if is_3D == 1:
			for p in range(0, Xtrain.shape[0]):
				Xtrain[p] = sc_X.fit_transform(Xtrain[p])
			for p in range(0, Xtest.shape[0]):
				Xtest[p] = sc_X.fit_transform(Xtest[p])
		elif is_3D == 0:
			Xtrain = (Xtrain - np.nanmean(Xtrain))/np.nanstd(Xtrain)
			Xtest = (Xtest - np.nanmean(Xtest))/np.nanstd(Xtest)
			#Xtrain = sc_X.fit_transform(Xtrain)
			#Xtest = sc_X.transform(Xtest)
		
		Xtrain_mean, Xtrain_std = np.nanmean(Xtrain), np.nanstd(Xtrain)
		Xtest_mean, Xtest_std = np.nanmean(Xtest), np.nanstd(Xtest)
		print("Xtrain_mean, std, Xtest_mean, std", Xtrain_mean, Xtrain_std, Xtest_mean, Xtest_std)
		
		Xtrain[np.isnan(Xtrain)]=0
		Xtest[np.isnan(Xtest)]=0
		Xtrain_mean, Xtrain_std = np.mean(Xtrain), np.std(Xtrain)
		Xtest_mean, Xtest_std = np.mean(Xtest), np.std(Xtest)
		print("Xtrain_mean, std, Xtest_mean, std", Xtrain_mean, Xtrain_std, Xtest_mean, Xtest_std)
		
		pdb.set_trace()
		
		return Xtrain, ytrain, Xtest, ytest  

	def xy_freq(self):
		x_simple = self.data[:,:-1]
		x_simple = x_simple.astype(float)
		y_simple = self.data[:,-1]
		print(x_simple.shape, y_simple.shape)
		Xtrain, ytrain, Xtest, ytest = self.xy_output(x_simple, y_simple, 0)
		return Xtrain, ytrain, Xtest, ytest
   
	def xy_simple(self):
		#new x_simple: one large array, flatten
		nr = self.data.shape[0] #pid
		nc = self.data.shape[1] * self.data.shape[2] #test/time
		#nc = nc + 2 #add encode gender col
		x_simple = np.zeros((nr, nc))
		y_simple = np.zeros((nr,1))
		x_pid = np.zeros((nr,1))
		i = 0 #thru pid
		for r in range(0,nr):
			print("pid", i)
			j = 0 #thru test
			k = 0 #thru time
			for c in range(0,nc): #change to nc-2 to include gender
				#only test results:
				x_simple[r,c] = self.data[i][j][k,2] #r0/t0..r0/t59, r1/t0	
				print("pid/test/time for row/col", i, j, k, "|", r, c)
				if k == (self.data.shape[2]-1): #fill all time per test
					j += 1 #next test
					k = 0 #reset time
				else:                    
					k += 1 #each col = next time same test    
			#pdb.set_trace()			           
			#x_simple[r, -2:] = self.data[i][0][0,[4,5]] #include gender col
			y_simple[r] = self.data[i][0][0,-1] #infection per pid
			x_pid[r] = self.data[i][0][0,0]
			i += 1 #each row = next pid
            
		pdb.set_trace()
		print(x_simple.shape, y_simple.shape)		
		Xtrain, ytrain, Xtest, ytest = self.xy_output(x_simple, y_simple, 0) #1=3D arr
		#np.save('xpid.npy', x_pid)
		return Xtrain, ytrain, Xtest, ytest

	def conv_xy_simple(self, test_num):
		#new conv_xy_simple (Npid, tsample) per test
		nr = self.data.shape[0] #pid
		nc = self.data.shape[2] #tSample
		x_simple = np.zeros((nr, nc))
		y_simple = np.zeros((nr,1))
		for r in range(0, nr):
			x_simple[r, :] = np.reshape(self.data[r][test_num][:,2], (1, nc))
			y_simple[r] = self.data[r][0][0,-1]
		
		print(x_simple.shape, y_simple.shape)
		Xtrain, ytrain, Xtest, ytest = self.xy_output(x_simple, y_simple, 0) #1=3D arr
		return Xtrain, ytrain, Xtest, ytest
	
