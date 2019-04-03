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

class frequency_bTest:
	"""
	Try1: assume informative missing, not missing at random, data = freq feature 
	Try2: concatenate freq feature with results (no need, try 1 good results)
	"""
	def __init__(self, xy):
		self.xy = np.array(xy)
	
	def create_freqTable(self):
		p = check_unique(self.xy[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		t = check_unique(self.xy[:,1]) #testType
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
		
		PIDfreq_list = []
		for r1, elem1 in enumerate(p_uTest):
			testfreq_list = []
			for r2, elem2 in enumerate(t_uTest):
				match = [x for x in list(self.xy) if x[1] == elem2 and x[0] == elem1]
				freq = len(match)
				testfreq_list.append(freq)
			#pdb.set_trace()
			pid_inf_idx = np.where(self.xy[:,0] == elem1)[0][0]
			pid_inf = self.xy[pid_inf_idx,-1]
			testfreq_list.append(pid_inf)
			PIDfreq_list.append(testfreq_list)
		return PIDfreq_list
		
		
