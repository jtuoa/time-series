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
REL_DAYRANGE = 60 #day prior knowledge


def check_unique(arr):
	u,c = np.unique(arr, return_counts=True)
	uc_table = np.asarray((u,c)).T
	print(uc_table)
	print("uc_table shape: ", uc_table.shape)
	return uc_table

class downSample_preprocess:
	"""
	ensemble classifier
	"""
	def __init__(self, x, y, has_inf, hasnot_inf):
		self.x = np.array(x)
		self.y = np.array(y)
		self.has_inf = has_inf
		self.hasnot_inf = hasnot_inf
	
	def train_test_split(self):
		x_hasinf = []
		x_hasnotinf = []
		for i in range(0, len(self.y)):
			if self.y[i] == 1:
				x_hasinf.append(self.x[i,:])
			else:
				x_hasnotinf.append(self.x[i,:])
		assert len(x_hasinf) == self.has_inf, "Error: incorrect has_inf"
		assert len(x_hasnotinf) == self.hasnot_inf, "Error: incorrect hasnot_inf"
		
		#pdb.set_trace()
		y_hasinf = [1] * int(self.has_inf)
		y_hasnotinf = [0] * int(self.hasnot_inf)
		
		Xtrain_hasinf, Xtest_hasinf, ytrain_hasinf, ytest_hasinf = train_test_split(x_hasinf, y_hasinf, test_size=0.2, random_state=42)
		
		hasnotinf_split = int(np.floor(self.hasnot_inf/len(Xtrain_hasinf)))
		x_hasnotinf_sets = []
		for i in range(0, hasnotinf_split):
			#random remove len(Xtrain_hasinf) number of samples from Xtest no replace
			indices = random.sample(range(0, len(x_hasnotinf)), len(Xtrain_hasinf))
			x_hasnotinf_sets.append([x_hasnotinf[i] for i in indices])
			for idx in sorted(indices, reverse=True):
				del x_hasnotinf[idx]		
			#pdb.set_trace()
		
		indices = random.sample(range(0, len(x_hasnotinf)), len(Xtest_hasinf))
		Xtest_hasnotinf = []
		Xtest_hasnotinf.append([x_hasnotinf[i] for i in indices])
		assert len(Xtest_hasnotinf[0]) == len(Xtest_hasinf), "Error: incorrect hasnotinf"
			
		#check length of x_hasnotinf_sets
		for i in range(0, len(x_hasnotinf_sets)):
			assert len(x_hasnotinf_sets[i]) == len(Xtrain_hasinf), "Error: incorrect hasnotinf_sets"
	
		return Xtrain_hasinf, Xtest_hasinf, x_hasnotinf_sets, Xtest_hasnotinf
		
	def train_validation_split(self, xtrain, ytrain):
		Xtrain, Xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
		return Xtrain, Xtest, ytrain, ytest
	

