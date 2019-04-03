import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.mlab as mlab
import random
import pdb
import copy

DAY_MIN = 1440
REL_DAYRANGE = 60 #day prior knowledge 60, 0

def check_unique(arr):
	u,c = np.unique(arr, return_counts=True)
	uc_table = np.asarray((u,c)).T
	print(uc_table)
	print("uc_table shape: ", uc_table.shape)
	return uc_table
	
	
class grid_preprocess:
	def __init__(self, xy):
		self.xy = np.array(xy)
		
	def sample_period_test(self):
		p = check_unique(self.xy[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		t = check_unique(self.xy[:,1]) #testType
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
			
		p_row = []
		for r1, elem1 in enumerate(p_uTest):
			p_meantest = []
			for r2, elem2 in enumerate(t_uTest):
				p_atest = [x for x in list(self.xy) if x[0] == elem1 and x[1] == elem2]
				p_time = np.array(sorted(p_atest, key=itemgetter(3), reverse=True))
				if len(p_time) == 0: #no test 
					p_meantest.append(0)
				else:
					p_time = p_time[:,3]
					'''p_dtime = []					
					p_dtime.append(p_time[0]-rel1)
					for i in range(0, len(p_time)-1):
						p_dtime.append(p_time[i+1] - p_time[i])
					p_meantest.append(np.mean(np.abs(p_dtime)))'''
					p_meantest.append(np.mean(np.abs(p_time)))
			#pdb.set_trace()
			p_row.append(p_meantest)
		p_row = np.array(p_row) #(856,20)
		
		#mean per test avg per col
		#pdb.set_trace()
		avg_alltest = []
		for i in range(0, p_row.shape[1]):
			avg_alltest.append(np.mean(p_row[:,i]))
		avg_alltest = np.array(avg_alltest)
		avg_alltest = avg_alltest / DAY_MIN #convert to min
		pdb.set_trace()
		return avg_alltest
		
	def classify_missing(self):
		p = check_unique(self.xy[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		t = check_unique(self.xy[:,1]) #testType
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
		
		#rel1 = REL_DAYRANGE * DAY_MIN #pos=before surgery
		#t_window = np.linspace(rel2, rel1, 11)
		
		p_row = []
		count_truemissing = 0
		count_lacksampling = 0
		for r1, elem1 in enumerate(p_uTest):
			for r2, elem2 in enumerate(t_uTest):
				p_atest = [x for x in list(self.xy) if x[0] == elem1 and x[1] == elem2]
				p_time = np.array(sorted(p_atest, key=itemgetter(3), reverse=True))
				if len(p_time) == 0: #no test 
					count_truemissing += 11
				else:
					p_time = p_time[:,3]
					for i in range(0, len(p_time)-1):
						if p_time[i+1] - p_time[i] >= 11520:
							count_truemissing += 1
						else:
							count_lacksampling += 1
		pdb.set_trace()
		return count_truemissing, count_lacksampling
					
	
