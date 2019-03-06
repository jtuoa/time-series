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


	
class impute_preprocess:
	"""
	pre-process for impute algorithms: Forward, PCA
	"""
	def __init__(self, xy, rel2, nSamples, method):
		self.xy = np.array(xy)
		self.nSamples = nSamples
		self.rel2 = rel2 #min
		self.method = method

	'''def dateWeightVector(self, impute, t_window):
		tnew_window = abs(np.array(t_window))
		surgDay = np.argmin(tnew_window)
		mu = 0
		sigma = 1
		weight = 2 #1: 0.96 0.98 1 vs. 2: 0.84 0.96 1
		samples = (surgDay * 2) + 1
		x = np.linspace(mu - weight*sigma, mu + weight*sigma, samples)
		s = mlab.normpdf(x, mu, sigma)
		s = s / max(s)
		s = s[0:len(t_window)]
		impute_arr = np.array(impute)
		output = np.multiply(np.array(impute_arr[:,2]), s)
		impute_arr[:,2] = output
		return list(impute_arr)'''
		
	#impute data prep
    #['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
	def impute_data(self):
		p = check_unique(self.xy[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		t = check_unique(self.xy[:,1]) #testType
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
		
		rel1 = REL_DAYRANGE * DAY_MIN #pos=before surgery 
		t_window = np.linspace(self.rel2, rel1, self.nSamples)
		t_window = sorted(t_window, reverse=True)
		
		PID_allTest = []
		check_total_rows = 0
		#count_impute = 0
		count_imputePID = []
        #for each PID
		for r1, elem1 in enumerate(p_uTest):
			allTest = []
			count_impPID = 0
			#search for all testA for this pid = match
			for r2, elem2 in enumerate(t_uTest):	
				match = [x for x in list(self.xy) if x[1] == elem2 and x[0] == elem1]
				check_total_rows += len(match)				
				#count impute
				#pdb.set_trace()
				#count_impute += self.nSamples - len(match)				
				#Empty: there is no testA for this pid
				if len(match) == 0:
					#pdb.set_trace()
					#find mean all pid based on inf
					pid_inf_idx = np.where(self.xy[:,0] == elem1)[0][0] #inf same for all test
					pid_inf = self.xy[pid_inf_idx,-1]
					mean = self.mean_perTest(elem2, pid_inf) 
					non_impute = self.create_nonImpute_list(pid_inf_idx, elem1, elem2, t_window, mean)
					#non_impute = self.dateWeightVector(non_impute, t_window) #mult weight
					count_impTest = self.nSamples
					allTest.append(non_impute)
				else:					
					#pdb.set_trace() 
					if self.method == 'LOCF':                   
						LOCF_impute, count_impTest = self.impute_LOCF_bTest(match, t_window)
						#LOCF_impute = self.dateWeightVector(LOCF_impute, t_window)
						allTest.append(LOCF_impute)
					elif self.method == 'NN':
						NN_impute, count_impTest = self.impute_NN_bTest(match, t_window)
						#NN_impute = self.dateWeightVector(NN_impute, t_window)
						allTest.append(NN_impute)
					elif self.method == 'PCA':
						PCA_impute = self.impute_PCA_bTest(match, t_window)
						#PCA_impute = self.dateWeightVector(PCA_impute, t_window)
						allTest.append(PCA_impute)
					elif self.method == 'MEAN':
						MEAN_impute, count_impTest = self.impute_MEAN_bTest(match, t_window)
						#MEAN_impute = self.dateWeightVector(MEAN_impute, t_window)
						allTest.append(MEAN_impute)
						
				count_impPID += count_impTest
			
			count_imputePID.append(count_impPID)
			#pdb.set_trace()
			#(PID, typeTests, Nsamples, attributes)
			PID_allTest.append(allTest)
			check = np.array(PID_allTest)
			print("                          ")
			print(check.shape)
			print("                          ")
		pdb.set_trace()
		np.save('LOCF_count_imputePID.npy', count_imputePID)
		check_unique_pid = check_unique(self.xy[:,0])
		print("total rows & unique pid:", check_total_rows, check_unique_pid)
		#print("count impute:", count_impute)
		return PID_allTest
	
	
	#Impute: Mean value of test depend on inf or not, map to closest val once
	def impute_MEAN_bTest(self, arr, t_window):
		arr = np.array(sorted(arr, key=itemgetter(3), reverse=True))
		pid_inf = arr[0,-1]
		testName = arr[0,1]	
		mean = self.mean_perTest(testName, pid_inf)
		t_window = np.array(t_window)
		slist_mean_impute = []
		start_idx = 0
		for arow, aval in enumerate(arr[:,3]):
			realClosestTime = 100e9
			val_idx = np.argmin(abs(t_window-aval)) #put val at nearest
			#pdb.set_trace()
			#fill val in-between with mean
			if start_idx <= val_idx:
				for i in range(start_idx, val_idx):
					stmp = list(arr[arow,:])
					stmp[2] = mean
					stmp[3] = t_window[i]
					slist_mean_impute.append(stmp)
			else: #if 2 within same interval replace with most recent
				#pdb.set_trace()
				del slist_mean_impute[len(slist_mean_impute)-1] #last item in list
			#pdb.set_trace()
			stmp = list(arr[arow,:])
			stmp[3] = t_window[val_idx]
			slist_mean_impute.append(stmp)
			start_idx = val_idx + 1
		#pdb.set_trace()
		listLen = len(slist_mean_impute)
		if listLen != len(t_window):
			for i in range(listLen, len(t_window)):
				stmp = list(arr[arow,:])
				stmp[2] = mean
				stmp[3] = t_window[i]
				slist_mean_impute.append(stmp)
		
		#pdb.set_trace()		
		tmp = np.array(slist_mean_impute)
		count_impTest = self.nSamples - len(set(tmp[:,2]))
		return slist_mean_impute, count_impTest
	
	#Impute forward: find time closest to sample point
	#LOCF = last observation carried forward
    #['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
	def impute_LOCF_bTest(self, arr, t_window):	
		arr = np.array(sorted(arr, key=itemgetter(3), reverse=True)) #sort by date
		t_window = np.array(t_window)
		slist_forward_impute = []
		for arow, aval in enumerate(arr[:,3]):
			if arow == 0: 
				idx = np.where((t_window >= aval))
				for i in range(0, len(idx[0])):
					stmp = list(arr[arow,:])
					stmp[3] = t_window[idx[0][i]]
					slist_forward_impute.append(stmp)
			else:
				idx = np.where((t_window >= aval) & (t_window < arr[arow-1,3])) 
				           
				for i in range(0, len(idx[0])):
					stmp = list(arr[arow-1,:]) #fill prev value
					stmp[3] = t_window[idx[0][i]]
					slist_forward_impute.append(stmp)
            
		#pdb.set_trace()		
		#if list unfilled, fill rest with last val
		listLen = len(slist_forward_impute)
		#print("slist Length ", listLen)   
		if listLen != len(t_window):
			#print("Enter fill function")          
			for i in range(listLen, len(t_window)):
				stmp = copy.deepcopy(list(arr[arow]))
				stmp[3] = t_window[i]
				slist_forward_impute.append(stmp)
		assert len(slist_forward_impute) == len(t_window), "Error: incorrect list length"
		#pdb.set_trace()
		tmp = np.array(slist_forward_impute)
		count_impTest = self.nSamples - len(set(tmp[:,2]))
		return slist_forward_impute, count_impTest

	#Impute 1NN: nearest neighbor
	def impute_NN_bTest(self, arr, t_window):
		arr = np.array(sorted(arr, key=itemgetter(3), reverse=True))
		slist_nn_impute = []
		for tval in t_window:
			realClosestTime = 100e9
			for arow, aval in enumerate(arr[:,3]):
				closestTimeCheck = abs(tval - aval)
				if closestTimeCheck < realClosestTime:
					realClosestTime = closestTimeCheck
					realIdx = arow
			#print("sample, arrIdx", tval, arr[realIdx,:]) 
			stmp = list(arr[realIdx,:])
			stmp[3] = tval
			slist_nn_impute.append(stmp)
			
		tmp = np.array(slist_nn_impute)
		count_impTest = self.nSamples - len(set(tmp[:,2]))
		return slist_nn_impute, count_impTest
		
    #Fill by: mean over all pid for pid with no measurements in testA
	def mean_perTest(self, test, has_inf):
		#compute mean across all patients for test based on pid infection
		match = [x for x in list(self.xy) if x[1] == test and x[-1] == has_inf]
		match_arr = np.array(match)
		if len(match_arr) == 0:
			mean = 0
		else:
			mean = np.sum(match_arr[:,2])/match_arr.shape[0]
		return mean

	#Create list: for PID that dont have any measurement in testA
    #['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
	def create_nonImpute_list(self, pid_idx, pid, test, t_window, mean):
		arr = np.empty([self.nSamples, self.xy.shape[1]], dtype=object)
		arr[:,0].fill(pid)
		arr[:,1].fill(test)
		arr[:,2].fill(mean)       
		arr[:,3] = t_window        
		arr[:,4:] = self.xy[pid_idx,4:] #gender, infection
		arr_list = list(arr)
		return arr_list





