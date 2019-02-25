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
		return Xtrain, Xtest		
			
		
		
class prepareData_MLP:
	def __init__(self, data):
		#old x: (856, 50, 60, 7) [PID, test_type, nSamples, col]
    	#['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
		self.data = np.array(data)

	def xy_output(self, x_simple, y_simple, is_3D):
		#pdb.set_trace()
		#stratify: preserves 21% ones and 79% zeros in y_simple
		Xtrain, Xtest, ytrain, ytest = train_test_split(x_simple, y_simple, test_size=0.2, random_state=42, stratify=y_simple) 
                
		sc_X = StandardScaler() #standardize data
		if is_3D == 1:
			for p in range(0, Xtrain.shape[0]):
				Xtrain[p] = sc_X.fit_transform(Xtrain[p])
			for p in range(0, Xtest.shape[0]):
				Xtest[p] = sc_X.fit_transform(Xtest[p])
		elif is_3D == 0:
			Xtrain = sc_X.fit_transform(Xtrain)
			Xtest = sc_X.transform(Xtest)	
        		
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
			i += 1 #each row = next pid
            
		pdb.set_trace()
		print(x_simple.shape, y_simple.shape)		
		Xtrain, ytrain, Xtest, ytest = self.xy_output(x_simple, y_simple, 0) #1=3D arr
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
			pid_inf_idx = np.where(self.xy[:,0] == elem1)[0][0]
			pid_inf = self.xy[pid_inf_idx,-1]
			testfreq_list.append(pid_inf)
			PIDfreq_list.append(testfreq_list)
		return PIDfreq_list
		
class impute_preprocess:
	"""
	pre-process for impute algorithms: Forward, PCA
	"""
	def __init__(self, xy, rel2, nSamples, method):
		self.xy = np.array(xy)
		self.nSamples = nSamples
		self.rel2 = rel2 #min
		self.method = method

	def dateWeightVector(self, impute, t_window):
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
		return list(impute_arr)
		
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
        #for each PID
		for r1, elem1 in enumerate(p_uTest):
			allTest = []
			#search for all testA for this pid = match
			for r2, elem2 in enumerate(t_uTest):	
				match = [x for x in list(self.xy) if x[1] == elem2 and x[0] == elem1]
				check_total_rows += len(match)
				#pdb.set_trace()
				#Empty: there is no testA for this pid
				if len(match) == 0:
					#pdb.set_trace()
					#find mean all pid based on inf
					pid_inf_idx = np.where(self.xy[:,0] == elem1)[0][0] #inf same for all test
					pid_inf = self.xy[pid_inf_idx,-1]
					mean = self.mean_perTest(elem2, pid_inf) 
					non_impute = self.create_nonImpute_list(pid_inf_idx, elem1, elem2, t_window, mean)
					non_impute = self.dateWeightVector(non_impute, t_window) #mult weight
					allTest.append(non_impute)
				else:					
					pdb.set_trace() 
					if self.method == 'LOCF':                   
						LOCF_impute = self.impute_LOCF_bTest(match, t_window)
						#LOCF_impute = self.dateWeightVector(LOCF_impute, t_window)
						allTest.append(LOCF_impute)
					elif self.method == 'NN':
						NN_impute = self.impute_NN_bTest(match, t_window)
						#NN_impute = self.dateWeightVector(NN_impute, t_window)
						allTest.append(NN_impute)
					elif self.method == 'PCA':
						PCA_impute = self.impute_PCA_bTest(match, t_window)
						#PCA_impute = self.dateWeightVector(PCA_impute, t_window)
						allTest.append(PCA_impute)
					elif self.method == 'MEAN':
						MEAN_impute = self.impute_MEAN_bTest(match, t_window)
						#MEAN_impute = self.dateWeightVector(MEAN_impute, t_window)
						allTest.append(MEAN_impute)
			#pdb.set_trace()
			#(PID, typeTests, Nsamples, attributes)
			PID_allTest.append(allTest)
			check = np.array(PID_allTest)
			print("                          ")
			print(check.shape)
			print("                          ")
		#pdb.set_trace()
		check_unique_pid = check_unique(self.xy[:,0])
		print("total rows & unique pid:", check_total_rows, check_unique_pid)
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
		return slist_mean_impute
	
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
		#print("fill slist Length ", len(slist_forward_impute))
		assert len(slist_forward_impute) == len(t_window), "Error: incorrect list length"
	
		#pdb.set_trace()
		return slist_forward_impute

	#Impute 1NN: nearest neighbor
	def impute_NN_bTest(self, arr, t_window):
		arr = np.array(sorted(arr, key=itemgetter(3), reverse=True))
		#print("input arr", arr)
		slist_nn_impute = []
		#pdb.set_trace()
		for tval in t_window:
			realClosestTime = 100e9
			for arow, aval in enumerate(arr[:,3]):
				closestTimeCheck = abs(tval - aval)
				if closestTimeCheck < realClosestTime:
					realClosestTime = closestTimeCheck
					realIdx = arow
			#pdb.set_trace()
			#print("sample, arrIdx", tval, arr[realIdx,:]) 
			stmp = list(arr[realIdx,:])
			stmp[3] = tval
			slist_nn_impute.append(stmp)
		#pdb.set_trace()
		return slist_nn_impute
		
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
    
class data_preprocess:
	def __init__(self, filename1, filename2, ntopTests, nYob, ntestRel_useRange):
		self.filename1 = filename1
		self.filename2 = filename2
		self.ntopTests = ntopTests
		self.nYob = nYob
		self.ntestRel_useRange = ntestRel_useRange
	
	def load_data(self):
		#load both datasets
		dataset1 = pd.read_csv(self.filename1, delimiter='\t', encoding='utf-8')
		dataset1.fillna(0, inplace = True)
		self.x1 = dataset1.loc[:,['PID', 'Date', 'TestType', 'NumAnswer']]
		self.x1 = np.array(self.x1) #(504675,4) 909 pid
		#self.x1 = self.x1[1580:1620,:] #test
		#self.x1 = self.x1[2010:2050,:] #test convert inf 2/0
		#self.x1 = self.x1[85944:85994,:] #test convert inf 2/1
		self.x1 = self.x1[0:104675,:]
		
		dataset2 = pd.read_csv(self.filename2, delimiter='\t', encoding='utf-8')
		dataset2.fillna(0, inplace = True)
		self.x2 = dataset2.loc[:,['PID', 'Infection', 't.Infection', 't.IndexSurgery', 'Sex', 'YoB']]
		self.x2 = np.array(self.x2)
		#pdb.set_trace()
	
	def keep_ntopTests(self):
		#top unique test elements: common blood tests more meaningful
		#use ntopTests = top blood test results
		u, c = np.unique(self.x1[:,2], return_counts=True) #get unique
		uc_table = np.asarray((u,c)).T
		testType_sort = sorted(uc_table, key=itemgetter(1), reverse=True)
		ntopTests_type = testType_sort[0:self.ntopTests] #get ntopUnique
		#x1 keep only top tests by append to x1new
		self.x1new = []
		for r1, elem1 in enumerate(self.x1):
			if(any(elem1[2] == r2[0] for r2 in ntopTests_type)):
				self.x1new.append(elem1)
		self.x1new = np.array(self.x1new) #(404803,4) 907 pid
		#pdb.set_trace()
	
	def combine_datasets(self):
        #combine x1 measurement, x2 metadata
		self.x1new = np.hstack((self.x1new, np.zeros((self.x1new.shape[0], 5), dtype=self.x1new.dtype)))
		for r1, elem1 in enumerate(self.x1new):
			r2 = list(self.x2[:,0]).index(elem1[0])
			self.x1new[r1,4:] = self.x2[r2,1:] #(404803,9) 907 pid
		#pdb.set_trace()
		
	def convert_infection2(self):
		self.x1new = pd.DataFrame(self.x1new, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.Infection', 't.IndexSurgery', 'Sex', 'YoB'])
		#self.x1new = self.x1new[self.x1new.Infection != 2] #del inf 2
		self.x1new.Infection = self.x1new.Infection.replace(2,1) #replace 2 with 1
		self.x1new = np.array(self.x1new) #(404803,9) 907 pid
		#pdb.set_trace()
	
	def create_surgery_test_dateCol(self): #col 9 shape(10)
		#create newDate col (rel minutes) = surgery date - test date
		self.x1new = np.hstack((self.x1new, np.zeros((self.x1new.shape[0], 1), dtype=self.x1new.dtype)))
		for i in range(0, len(self.x1new)):
			surgery_time = datetime.strptime(self.x1new[i,6], '%Y-%m-%d %H:%M:%S')
			test_time = datetime.strptime(self.x1new[i,1], '%d/%m/%y %H:%M')
			surgery_test_time = surgery_time - test_time
			#store relative minutes
			if surgery_test_time.days < 0:
				surgery_test_sec = -1 * surgery_test_time.seconds
			else:
				surgery_test_sec = +1 * surgery_test_time.seconds
			#pos = test before surgery, neg = test after surgery			
			self.x1new[i,-1] = (surgery_test_time.days*DAY_MIN) + surgery_test_sec/S_MIN
		#pdb.set_trace() #(404803, 10) 907 pid

	def create_surgery_infection_dateCol(self): #col 10: only to extract minInfTime shape(11)
		#create newDate col (rel min) = surgery date - infectiondate
		self.x1new = np.hstack((self.x1new, np.zeros((self.x1new.shape[0], 1), dtype=self.x1new.dtype)))
		searchMaxInfTime = []
		for i in range(0, len(self.x1new)):
			if self.x1new[i,5] == 0: #handle no infection, time = 0
				#pdb.set_trace()
				self.x1new[i,-1] = 0
			else:				
				surgery_time = datetime.strptime(self.x1new[i,6], '%Y-%m-%d %H:%M:%S')
				inf_time = datetime.strptime(self.x1new[i,5], '%Y-%m-%d %H:%M:%S')
				surgery_inf_time = surgery_time - inf_time
				searchMaxInfTime.append(surgery_inf_time)
				#store relative minutes
				if surgery_inf_time.days < 0:
					surgery_inf_sec = -1 * surgery_inf_time.seconds
				else:
					surgery_inf_sec = +1 * surgery_inf_time.seconds
				tmp = (surgery_inf_time.days*DAY_MIN) + surgery_inf_sec/S_MIN
				#check can infection happen on surgery dat: count_tmp=0 not possible
                #if tmp == 0: count_tmp += 1
				#pos = inf before surgery, neg = inf after surgery			
				self.x1new[i,-1] = tmp #(404803, 11) 907 pid
				#print("TIME ", i, self.x1new[i,10])
		#pdb.set_trace()
		minInfTime = min(searchMaxInfTime) 
		self.rel2 = (minInfTime.days*DAY_MIN) - (minInfTime.seconds/S_MIN) #min
	
	def discard_testRelRange(self):
		if self.ntestRel_useRange:
			self.xnew = []
			patient_drop = []
			rel1 = REL_DAYRANGE * DAY_MIN #pos=before surgery
			rel2 = self.rel2 #keep blood tests up to furthest infection time, neg=after surgery
			for r1, elem1 in enumerate(self.x1new):
				if(elem1[9] <= rel1 and elem1[9] >= rel2): #time window
					self.xnew.append(elem1)
				else:
				    patient_drop.append(elem1)

			#pdb.set_trace()
			self.xnew=np.array(self.xnew) #(64473, 11) 856 pid
		
		#self.xnew = copy.deepcopy(self.x1new) #for dataSparsity plot all           
		#pdb.set_trace()
	
	def encode_gender(self): #col: 11,12 shape(13)
		label_encoder = LabelEncoder()
		gender_integer_encoded = label_encoder.fit_transform(self.xnew[:,7])
		gender_onehot_encoded = to_categorical(gender_integer_encoded)
		self.xnew = np.hstack((self.xnew, gender_onehot_encoded)) #(64473,13) 856 pid
		#pdb.set_trace()

	def integer_encode_testName(self):
		#pid, testDate, testType, numAns, inf, infDate, surgDate, gen, yob, surg_test_date, surg_inf_date
		u, c = np.unique(self.xnew[:,2], return_counts=True) #get unique
		uc_table = np.asarray((u,c)).T
		self.xnew = np.array(sorted(self.xnew, key=itemgetter(2)))
		self.testName_dict = dict(list(enumerate(uc_table[:,0])))
		idx1 = 0
		for i in range(0, len(uc_table)):
			idx2 = uc_table[i,1] + idx1
			self.xnew[idx1:idx2,2] = i
			idx1 = idx2
		#pdb.set_trace()            
		return self.xnew #(64473,13) 856 pid

	def result(self):
		#pdb.set_trace()
		check = check_unique(self.xnew[:,0])
		print(check) #856 pid
		
		testName = check_unique(self.xnew[:,2])
		testName = testName[:,0] #50
				
		#new dataset
		dfxy_all = pd.DataFrame(self.xnew, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.Infection', 't.IndexSurgery', 'Sex', 'YoB', 'surgery_test_date', 'surgery_inf_date', 'Gender_encode0', 'Gender_encode1'])
		dfxy = dfxy_all.loc[:,['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']]
		
		return dfxy, self.rel2, self.testName_dict




