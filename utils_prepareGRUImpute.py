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

class prepareData_GRUD:
	"""
	create/combine x, mask and time interval vector
	put value at time interval
	"""
	def __init__(self, xy, ntopTests):
		self.xyold = np.array(xy)
		self.ntopTests = ntopTests
	
	def upSample(self, has_inf, hasnot_inf):
		#pdb.set_trace()
		i_hasinf = np.where(self.xyold[:,-1]==1)[0]
		xy_inf = self.xyold[i_hasinf, :]
		num_inf = check_unique(xy_inf[:,0])
		assert num_inf.shape[0] == has_inf, "Error: incorrect number of infection1"
		
		N = hasnot_inf - has_inf
		pid_inf = num_inf[:,0]		
		pid_hasinfus = np.random.choice(pid_inf, size=int(N), replace=True)
		xy_us = []
		for i in range(0, N):
			print("upsampled pid", i)
			xypid_ustmp = [x for x in list(xy_inf) if x[0] == pid_hasinfus[i]]
			xypid_us = copy.deepcopy(xypid_ustmp)
			#pdb.set_trace()
			noise = np.random.normal(0, 0.05, len(xypid_us))
			#indicator of dup pid by *1000 since can't use string
			for j in range(0, len(xypid_us)):
				xypid_us[j][0] = xypid_us[j][0]*(1000+i)
				xypid_us[j][2] = xypid_us[j][2] + noise[j]
				
			#pdb.set_trace()
			xy_us.append(xypid_us)
		#pdb.set_trace()
		
		xy_arrus = np.array(xy_us[0]) #upsample arr
		for i in range(1, N):
			tmp = np.array(xy_us[i])
			print("tmp, arr", len(tmp), xy_arrus.shape)
			xy_arrus = np.vstack([xy_arrus, tmp])
		
		#pdb.set_trace()		
		xy_arrus = np.vstack([xy_arrus, xy_inf]) #us + orig_inf
		num_inf = check_unique(xy_arrus[:,0])
		assert num_inf.shape[0] == hasnot_inf, "Error: incorrect number of infection2"
		
		i_hasnotinf = np.where(self.xyold[:,-1]==0)[0]
		xy_arrus = np.vstack([xy_arrus, self.xyold[i_hasnotinf, :]])
		num_inf_noinf = check_unique(xy_arrus[:,0])
		assert num_inf_noinf.shape[0] == hasnot_inf + hasnot_inf, "Error: incorrect total"
		pdb.set_trace()
		self.xy = xy_arrus
		
	def time_watcher(self, arr, start_idx):
		#return indices of changes in time
		run = True
		end_idx = start_idx + 1
		while(run): #find range of test
			if end_idx == len(arr):
				run = False
			elif arr[end_idx][1] == arr[start_idx][1]:
				end_idx += 1
			else:
				#pdb.set_trace()
				run = False
		return start_idx, end_idx
	
	def append_nans(self, length, list_):
		diff_len = length - len(list_)
		if diff_len < 0:
			raise AttributeError("Length error list too long")
		return list_ + [np.nan] * diff_len
	
	def find_most_frequent(self, p_uTest):
		dicts_allpid = []
		for r1, elem1 in enumerate(p_uTest):
			#pdb.set_trace()
			match = [x for x in list(self.xy) if x[0] == elem1]
			match_time = np.array(sorted(match, key=itemgetter(3), reverse=True))
			match_test = np.array(sorted(match_time, key=itemgetter(1), reverse=False))
			
			#create dict per pid = {testnum:frequency}
			testNum = match_test[0,1]
			count = 1
			dicts = {}			
			if len(match_test) == 1:
				dicts[testNum] = count
			else:
				for i in range(1, len(match_test)):
					if testNum == match_test[i,1]:
						count += 1
						dicts[testNum] = count
					else:
						dicts[testNum] = count
						testNum = match_test[i,1]
						count = 1
			dicts_allpid.append(dicts)
			print("dicts_allpid", r1)
			#pdb.set_trace()
		#check if any empty dicts
		for i in range(0, len(dicts_allpid)):
			if len(dicts_allpid[i]) == 0:
				print("found empty dicts, stop")
				pdb.set_trace() 
		
		#pdb.set_trace()
		#create dict per pid = {testnum: largest freq}		
		dicts_longestallpid = []
		for i in range(0, len(dicts_allpid)):
			#first encountered largest val
			print("dicts_longestallpid", i)
			key, val = max(dicts_allpid[i].items(), key=lambda x:x[1])
			dicts_longestperpid = {}
			dicts_longestperpid[key] = val
			dicts_longestallpid.append(dicts_longestperpid)
		
		#pdb.set_trace()
		#find longest result of all pid
		max_val = []
		for i in range(0, len(dicts_longestallpid)):
			max_val.append(max(dicts_longestallpid[i].values()))

		#pdb.set_trace()
		#max_idx = pid, max_val = freq
		max_idx, max_val = max(enumerate(max_val), key=lambda p: p[1])
		max_testnum = dicts_longestallpid[max_idx].keys()
		max_testnum = list(max_testnum)[0]		
		return max_idx, max_val, max_testnum
	
	
	def split_standardize(self, x, y, is_nan):
		#stratify: preserves 21% ones and 79% zeros in y
		indices = np.arange(len(y))
		Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = train_test_split(x, y, indices, test_size=0.2, random_state=42, stratify=y) 
			     
		sc_X = StandardScaler() #standardize data
		if is_nan == 1: #some_nan
			for i in range(0, len(Xtrain)):
				Xtrain[i] = np.array(Xtrain[i])
				if np.isnan(np.nanmin(Xtrain[i])): #all_nan
					print("replace all with 0") #replace nan with 0
					try:
						Xtrain[i] = np.nan_to_num(Xtrain[i])
						pass #dont standardize
					except:
						print("list issue: all nan")
						pdb.set_trace()
				elif np.nanstd(Xtrain[i]) == 0: #same/one number
					print("same/one number", i)
					try:
						if np.any(Xtrain[i] == 0): #all zero
							print("avoid division by 0", i)
							pass
						else:
							Xtrain[i] = Xtrain[i] / Xtrain[i]
							#Xtrain[i] = (Xtrain[i] - np.nanmean(Xtrain[i]))/np.nanstd(Xtrain[i])
							Xtrain[i] = np.nan_to_num(Xtrain[i])
					except:
						print("list issue: same/one number")
						pdb.set_trace()
				else:
					try:				
						print("standardize", i)
						Xtrain[i] = (Xtrain[i] - np.nanmean(Xtrain[i]))/np.nanstd(Xtrain[i])
						Xtrain[i] = np.nan_to_num(Xtrain[i])
					except:
						print("standardize issue")
						pdb.set_trace()
			
			pdb.set_trace()	
			for i in range(0, len(Xtest)):
				Xtest[i] = np.array(Xtest[i])
				if np.isnan(np.nanmin(Xtest[i])):
					Xtest[i] = np.nan_to_num(Xtest[i])
					pass #dont standardize
				elif np.nanstd(Xtest[i]) == 0:
					if np.any(Xtest[i] == 0):
						pass
					else:
						Xtest[i] = Xtest[i] / Xtest[i]
						#Xtest[i] = (Xtest[i] - np.nanmean(Xtest[i]))/np.nanstd(Xtest[i])
						Xtest[i] = np.nan_to_num(Xtest[i])
				else:
					Xtest[i] = (Xtest[i] - np.nanmean(Xtest[i]))/np.nanstd(Xtest[i])
					Xtest[i] = np.nan_to_num(Xtest[i])
					
		elif is_nan == 2: 
			print("mask don't standardize")
			pass		
		
		return Xtrain, ytrain, Xtest, ytest
		
	def create_x(self):
		#get all test for patient
		p = check_unique(self.xy[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		#get most frequent result
		max_idx, max_val, max_testnum = self.find_most_frequent(p_uTest)		
		most_freq = [x for x in list(self.xy) if x[0] == p_uTest[max_idx] and x[1] == max_testnum]
		most_freq = np.array(sorted(most_freq, key=itemgetter(3), reverse=True))
		
		#start create x
		self.x = []
		self.y = []
		self.time = []
		for r1, elem1 in enumerate(p_uTest):
			#pdb.set_trace()
			print("patient  %d/1346" % r1)
			match = [x for x in list(self.xy) if x[0] == elem1]
			match_time = np.array(sorted(match, key=itemgetter(3), reverse=True))
			match_test = np.array(sorted(match_time, key=itemgetter(1), reverse=False))
						
			start_idx = 0
			end_idx = 0
			x = []
			time = []
			count_test = 0
			while(end_idx != len(match_test)):
				#pdb.set_trace()
				start_idx, end_idx = self.time_watcher(match_test, start_idx)
				#handle available data
				test_row=[]
				time_row=[]
				nan_startidx = 0
				for i in range(start_idx, end_idx):
					#append nans before/after/between points
					nan_endidx = np.argmin(abs(most_freq[:,3]-match_test[i,3]))			 
					remain_i = end_idx - i
					remain_nan = len(most_freq) - nan_endidx
					if remain_i > remain_nan:
						#pdb.set_trace()
						remain_diff = remain_i - remain_nan
						nan_endidx = nan_endidx - (remain_diff*4) #tune factor			
					for j in range(nan_startidx, nan_endidx):
						test_row.append(np.nan)
						time_row.append(most_freq[j,3])
						
					test_row.append(match_test[i,2])
					time_row.append(match_test[i,3])
					
					if nan_endidx <= nan_startidx:
						nan_startidx = nan_startidx + 1
					else:
						nan_startidx = nan_endidx + 1			
				
				#pdb.set_trace()
				#fill missing results at end
				if len(test_row) != len(most_freq):
					diff = len(most_freq) - len(test_row)
					idx = len(test_row)
					for k in range(0, diff):
						test_row.append(np.nan)
						try:
							time_row.append(most_freq[idx,3])
							idx += 1
						except:
							print("axis out of range")
							pdb.set_trace()
				
				#pdb.set_trace()
				#check for missing tests in-between/before and fill with np.nan
				if match_test[i,1] == count_test:
					count_test += 1
				else:
					for j in range(count_test, match_test[i,1]):
						print("have missing test in pid", count_test)
						nans = [np.nan]*len(most_freq)
						x.append(nans)

						time.append(most_freq[:,3])
						count_test += 1
					count_test += 1
				#pdb.set_trace()	
				x.append(test_row) #x[test]=results
				time_row = np.array(time_row)
				time.append(time_row)
				start_idx = end_idx
			
			#pdb.set_trace()
			#check for missing tests at the end and fill with 1 np.nan
			if count_test != 20:
				diff = 20 - count_test
				for i in range(0, diff):
					nans = [np.nan]*len(most_freq)
					x.append(nans)
					
					time.append(most_freq[:,3])
					count_test += 1
							
			#pdb.set_trace()
			self.x.append(x) #x=pid matrix, (pid, tests, time)
			self.time.append(time) #(pid, tests, time)
			self.y.append(match_test[0][6])			
			
		for i in range(0, len(self.x)):
			tmp = np.array(self.x[i])
			tmp2 = np.array(self.time[i])
			assert tmp.shape[0] == 20, "Error: not 20 x rows"
			assert tmp2.shape[0] == 20, "Error: not 20 time rows"
				
		
		#collapse each matrix pid into 1x1440
		'''pdb.set_trace()
		for pid in range(0, len(xlist)):
			xarr = np.array(xlist[pid][0])
			for test in range(1, len(xlist[0])):
				tmp = np.array(xlist[pid][test])
				try:
					xarr = np.vstack([xarr, tmp])
				except:
					pdb.set_trace()
					print("concatenate error")
			#pdb.set_trace()
			print("xarr shape", xarr.shape)
			#xarr = xarr.flatten()
			self.x.append(xarr)'''		
		
		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.x, self.y, is_nan=1)	
		return Xtrain, ytrain, Xtest, ytest #row = a pid = a batch
		
	
	def create_mask(self):
		self.mask=[]
		for i in range(0, len(self.x)):
			results = self.x[i]
			mask = np.isnan(results)
			mask = mask * 1
			mask = 1 - mask
			self.mask.append(mask)
		
		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.mask, self.y, is_nan=2)
		return Xtrain, Xtest		

	def create_delta(self):
		#collapse each matrix pid into 1x1440
		'''self.time_flatten = []
		for pid in range(0, len(self.time)):
			darr = np.array(self.time[pid][0])
			for test in range(1, len(self.time[0])):
				tmp = np.array(self.time[pid][test])
				try:
					darr = np.vstack([darr, tmp])
				except:
					print("concatenate error")
					pdb.set_trace()
			#pdb.set_trace()
			print("darr shape", darr.shape)
			#darr = darr.flatten()
			self.time_flatten.append(darr)'''
		
		#pdb.set_trace()
		self.delta = []
		time_len = len(self.x[0][0]) #72
		for pid in range(0, len(self.time)):
			delta=[]
			#pdb.set_trace()
			for test in range(0, self.ntopTests):
				#time_start = test * time_len
				#time_end = time_start + time_len
				delta_row = []
				for time in range(0, int(time_len)):
					print("pid= %d, test= %d, time= %d" % (pid, test, time))
					if time == 0:
						dt = 0
						delta_row.append(dt)
					else:
						if self.mask[pid][test][time-1] == 1:
							try:
								dt = self.time[pid][test][time] - self.time[pid][test][time-1]
							except:
								print("index problem")
								pdb.set_trace()
							delta_row.append(dt)
						elif self.mask[pid][test][time-1] == 0:
							try:
								dt = self.time[pid][test][time] - self.time[pid][test][time-1] + delta_row[-1]
							except:
								print("index problem")
								pdb.set_trace()
							delta_row.append(dt) #each test
				#pdb.set_trace()
				delta.append(delta_row) #each pid
			#pdb.set_trace()
			delta = np.array(delta)
			self.delta.append(delta)
			
		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.delta, self.y, is_nan=1)
		return Xtrain, Xtest
			
	def create_input(self, Xtrain_GRUDX, Xtest_GRUDX, Xtrain_GRUDM, Xtest_GRUDM, Xtrain_GRUDD, Xtest_GRUDD):
		#create long vectors by concatenate rows from [x,m,d]
		xmdtrain=[]
		xmdtest=[]
		for i in range(0, len(Xtrain_GRUDX)):
			xmdtrain_row = np.vstack([Xtrain_GRUDX[i], Xtrain_GRUDM[i], Xtrain_GRUDD[i]])
			xmdtrain.append(xmdtrain_row)
			
		for i in range(0, len(Xtest_GRUDX)):
			xmdtest_row = np.vstack([Xtest_GRUDX[i], Xtest_GRUDM[i], Xtest_GRUDD[i]])
			xmdtest.append(xmdtest_row)
		pdb.set_trace()
		return xmdtrain, xmdtest

#####################################################################################3
	'''def create_deltaOLD(self):
		pdb.set_trace()
		self.delta=[]
		for pid in range(0, len(self.time)):
			delta=[]
			for test in range(0, self.ntopTests):
				delta_row=[]
				for time in range(0, len(self.time[pid][0])): #length 1 !! incorr
					if time == 0:
						dt = 0
						delta_row.append(dt)
					else:
						if self.mask[pid][test][time-1] == 1:
							dt = self.time[pid][test][time]-self.time[pid][test][time-1]
							delta_row.append(dt)
						elif self.mask[pid][test][time-1] == 0:
							dt = self.time[pid][test][time]-self.time[pid][test][time-1]+delta_row[-1]
							delta_row.append(dt)
				#pdb.set_trace()
				delta.append(delta_row)
			#pdb.set_trace()
			self.delta.append(delta)	
		
		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.delta, self.y, is_nan=1)
		return Xtrain, Xtest'''	

	'''def split_standardizeOLD(self, x, y, is_nan):
		#stratify: preserves 21% ones and 79% zeros in y
		indices = np.arange(len(y))
		#make (patient, time, tests)
		xnew=[]
		for i in range(0, len(x)):
			tmp = np.array(x[i])
			tmp = np.transpose(tmp) #ROW=time, COL=fix num tests
			print("split shape", tmp.shape)
			xnew.append(tmp)

		pdb.set_trace()
		xnew = np.array(xnew)
		Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = train_test_split(xnew, y, indices, test_size=0.2, random_state=42, stratify=y) 
			     
		sc_X = StandardScaler() #standardize data
		if is_nan == 1: #some_nan
			for i in range(0, Xtrain.shape[0]):
				if np.isnan(np.nanmin(Xtrain[i])): #all_nan
					#pdb.set_trace()
					print("replace all with 0") #replace nan with 0
					tmp = Xtrain[i]
					tmp[np.isnan(tmp)]=0
					pass #dont standardize
				elif np.nanstd(Xtrain[i]) == 0: #same number
					print("same number", i)
					#pdb.set_trace()
					if np.any(Xtrain[i] == 0): #all zero
						print("avoid division by 0", i)
						pass
					else:
						Xtrain[i] = Xtrain[i] / Xtrain[i]
				else:
					#pdb.set_trace()				
					print("standardize", i)
					Xtrain[i] = (Xtrain[i] - np.nanmean(Xtrain[i]))/np.nanstd(Xtrain[i])
					tmp = Xtrain[i]
					tmp[np.isnan(tmp)]=0
				
			for i in range(0, Xtest.shape[0]):
				if np.isnan(np.nanmin(Xtest[i])):
					tmp=Xtest[i]
					tmp[np.isnan(tmp)]=0
					pass #dont standardize
				elif np.nanstd(Xtest[i]) == 0:
					if np.any(Xtest[i] == 0):
						pass
					else:
						Xtest[i] = Xtest[i] / Xtest[i]
				else:
					Xtest[i] = (Xtest[i] - np.nanmean(Xtest[i]))/np.nanstd(Xtest[i])
					tmp=Xtest[i]
					tmp[np.isnan(tmp)]=0
					
		elif is_nan == 2: 
			print("mask don't standardize")
			pass		
		
		return Xtrain, ytrain, Xtest, ytest'''
	

	'''def create_xOLD(self):
		#get all test for patient
		p = check_unique(self.xy[:,0]) #pid
		p_arr = np.array(p)
		
		self.x = []
		self.y = []
		self.time = []
		for r1, elem1 in enumerate(p_uTest):
			pdb.set_trace()
			match = [x for x in list(self.xy) if x[0] == elem1]
			match_time = np.array(sorted(match, key=itemgetter(3), reverse=True))
			match_test = np.array(sorted(match_time, key=itemgetter(1), reverse=False))
			print(match_test)
			
			start_idx = 0
			end_idx = 0
			x = []
			time = []
			count_test = 0
			while(end_idx != len(match_test)):
				#pdb.set_trace()
				start_idx, end_idx = self.time_watcher(match_test, start_idx)
				test_row=[]
				time_row=[]
				for i in range(start_idx, end_idx):
					test_row.append(match_test[i,2])
					time_row.append(match_test[i,3])				
				
				#pdb.set_trace()
				#check for missing tests in-between and fill with 1 np.nan
				if match_test[i,1] == count_test:
					count_test += 1
				else:
					for j in range(count_test, match_test[i,1]):
						print("have missing test in pid", count_test)
						test_rowNan=[]
						test_rowNan.append(np.nan)
						x.append(test_rowNan)
						
						time_rowNan=[]
						time_rowNan.append(np.nan)
						time.append(time_rowNan)
						count_test += 1
					count_test += 1
					
				x.append(test_row)
				time.append(time_row)
				start_idx = end_idx
			
			#check for missing tests at the end and fill with 1 np.nan
			#pdb.set_trace()
			if count_test != 20:
				diff = 20 - count_test
				for i in range(0, diff):
					test_rowNan=[]
					test_rowNan.append(np.nan)
					x.append(test_rowNan)
					
					time_rowNan=[]
					time_rowNan.append(np.nan)
					time.append(time_rowNan)
					count_test += 1
							
			#pdb.set_trace()
			#fill nan in time so equal col per patient
			fill_len = max(len(l) for l in x)
			for i in range(0, len(x)):
				x[i] = self.append_nans(fill_len, x[i])
				time[i] = self.append_nans(fill_len, time[i])			
			self.x.append(x)
			self.time.append(time)
			
			for i in range(0, len(self.x)): #debug: get 20 rows
				tmp = np.array(self.x[i])
				tmp2 = np.array(self.time[i])
				print("shape", tmp.shape, tmp2.shape)
			print("                           ")
				
			self.y.append(match_test[0][6])
		
	
		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.x, self.y, is_nan=1)	
		return Xtrain, ytrain, Xtest, ytest #feed rows into GRU'''
	
#######################################################################################


'''class prepareData_GRUDold:
	"""
	create/combine x, mask and time interval vector
	put value to nearest location and fill rest with NA
	"""
	def __init__(self, xy, rel2, nSamples, ntopTests):
		self.xy = np.array(xy)
		self.rel2 = rel2
		self.nSamples =10
		self.ntopTests = ntopTests
	
	def ll_length(self, l):
		length = []
		for i in range(0, len(l)):
			length.append(len(l[i]))
			#print("check length", i, len(l[i]))
		return max(length)
	
	def split_standardize(self, x, y, is_nan):
		#stratify: preserves 21% ones and 79% zeros in y
		indices = np.arange(len(y))
		Xtrain, Xtest, ytrain, ytest, train_indices, test_indices = train_test_split(x, y, indices, test_size=0.2, random_state=42, stratify=y) 
		
		Xtrain = np.array(Xtrain)
		Xtest = np.array(Xtest)
		     
		sc_X = StandardScaler() #standardize data
		if is_nan == 0:
			for i in range(0, Xtrain.shape[0]):
				Xtrain[i] = sc_X.fit_transform(Xtrain[i])
			for i in range(0, Xtest.shape[0]):
				Xtest[i] = sc_X.transform(Xtest[i])
		elif is_nan == 1:
			for i in range(0, Xtrain.shape[0]):
				if np.isnan(np.nanmin(Xtrain[i])):
					pdb.set_trace()
					print("dont standardize")
					pass #dont standardize
				elif np.nanstd(Xtrain[i]) == 0: #same number
					print("same numer", i)
					Xtrain[i] = Xtrain[i] / Xtrain[i]
				else:				
					print("standardize", i)
					Xtrain[i] = (Xtrain[i] - np.nanmean(Xtrain[i]))/np.nanstd(Xtrain[i])
			for i in range(0, Xtest.shape[0]):
				if np.isnan(np.nanmin(Xtest[i])):
					pass #dont standardize
				elif np.nanstd(Xtest[i]) == 0:
					Xtest[i] = Xtest[i] / Xtest[i]
				else:
					Xtest[i] = (Xtest[i] - np.nanmean(Xtest[i]))/np.nanstd(Xtest[i])
		elif is_nan == 2:
			print("mask don't standardize")
			pass		
		
		return Xtrain, ytrain, Xtest, ytest
	
	def create_xmt(self):
		#other schemes: each test has its own sampling scheme
		#all pid share same time sampling scheme: place results once at nearest
		rel1 = REL_DAYRANGE * DAY_MIN
		t_window = np.linspace(self.rel2, rel1, self.nSamples)
		t_window = sorted(t_window, reverse=True)
		t_window = np.array(t_window)
		
		p = check_unique(self.xy[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]

		t = check_unique(self.xy[:,1]) #testType
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
		
		self.x_pid = []
		self.y_pid = []
		for r1, elem1 in enumerate(p_uTest):
			x_row = []
			for r2, elem2 in enumerate(t_uTest):
				match = [x for x in list(self.xy) if x[0] == elem1 and x[1] == elem2]
				x_rowElem = []
				if len(match) == 0:
					match_pid = [x for x in list(self.xy) if x[0] == elem1]
									
					for i in range(0, len(t_window)):
						stmp = list(match_pid[0])
						stmp[1] = elem2
						stmp[2] = np.nan
						stmp[3] = t_window[i]
						#print("what is stmp", stmp)
						#pdb.set_trace()
						x_rowElem.append(stmp)
				else:
					match = np.array(sorted(match, key=itemgetter(3), reverse=True))
					start_idx = 0
					for arow, aval in enumerate(match[:,3]): #put val at nearest once
						realClosestTime = 100e9
						val_idx = np.argmin(abs(t_window-aval))
						#fill in between with nan
						if start_idx <= val_idx:
							for i in range(start_idx, val_idx):
								stmp = list(match[arow,:])
								stmp[2] = np.nan
								stmp[3] = t_window[i]
								x_rowElem.append(stmp)
						else: #if 2 within same interval replace with most current
							del x_rowElem[len(x_rowElem)-1]
						#pdb.set_trace()
						#fill val
						stmp = list(match[arow,:])
						stmp[3] = t_window[val_idx]
						x_rowElem.append(stmp)
						start_idx = val_idx + 1

					#pdb.set_trace()
					#print("test=", elem2)
					listLen = len(x_rowElem)
					if listLen != len(t_window): #fill rest of row with nan
						for i in range(listLen, len(t_window)):
							stmp = list(match[arow,:])
							stmp[2] = np.nan
							stmp[3] = t_window[i]
							x_rowElem.append(stmp)
				
				#pdb.set_trace()	
				x_row.append(x_rowElem)
				print("x_row length", len(x_row))
			#pdb.set_trace()
			self.y_pid.append(x_rowElem[0][-1])		
			self.x_pid.append(x_row)
			print("x_pid: ", r1)
			print("                  ")
				
	def create_x(self):
		self.createx_pid = []
		for pid in range(0, len(self.x_pid)):
			createx_row = []
			for test in range(0, self.ntopTests):
				results = np.array(self.x_pid[pid][test])
				createx_row.append(list(results[:,2]))
			#pdb.set_trace()
			self.createx_pid.append(createx_row)

		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.createx_pid, self.y_pid, is_nan=1)
		
		return Xtrain, ytrain, Xtest, ytest
		
		
	def create_mask(self):
		self.createmask_pid = []
		for pid in range(0, len(self.x_pid)):
			results = self.createx_pid[pid]
			mask = np.isnan(results)
			mask = mask * 1
			mask = 1 - mask
			self.createmask_pid.append(mask)
		
		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.createmask_pid, self.y_pid, is_nan=2)
		return Xtrain, Xtest
	
	def create_time(self):
		rel1 = REL_DAYRANGE * DAY_MIN
		t_window = np.linspace(self.rel2, rel1, self.nSamples)
		t_window = sorted(t_window, reverse=True)
		t_window = np.array(t_window)
		self.createtime_pid = []
		#pdb.set_trace()
		for pid in range(0, len(self.x_pid)):
			time_all = []
			for test in range(0, self.ntopTests):
				mask_row = self.createmask_pid[pid][test]
				time_row = []
				for time in range(0, self.nSamples):
					if time == 0:
						dt = 0
						time_row.append(dt)
					else:
						if mask_row[time-1] == 1:
							dt = t_window[time] - t_window[time-1]
							time_row.append(dt)
						elif mask_row[time-1] == 0:
							dt = t_window[time] - t_window[time-1] + time_row[-1]
							time_row.append(dt)
				#pdb.set_trace()
				time_all.append(time_row)
			#pdb.set_trace()
			self.createtime_pid.append(time_all)
		
		pdb.set_trace()
		Xtrain, ytrain, Xtest, ytest = self.split_standardize(self.createtime_pid, self.y_pid, is_nan=0)
		return Xtrain, Xtest	'''	

