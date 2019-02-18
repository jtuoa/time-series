import numpy as np
import pandas as pd
from operator import itemgetter
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
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


class prepareData_MLP:
	def __init__(self, data):
		#old x: (856, 50, 60, 7) [PID, test_type, nSamples, col]
    	#['PID', 'TestType', 'NumAnswer', 'surgery_test_date', 'Gender_encode0', 'Gender_encode1', 'Infection']
		self.data = np.array(data)

	def xy_output(self, x_simple, y_simple, is_3D):
		#pdb.set_trace()
		#stratify: preserves 21% ones and 79% zeros in y_simple
		Xtrain, Xtest, ytrain, ytest = train_test_split(x_simple, y_simple, test_size=0.2, random_state=0, stratify=y_simple) 
        
		sc_X = StandardScaler() #standardize data
		if is_3D:
			for p in range(0, Xtrain.shape[0]):
				Xtrain[p] = sc_X.fit_transform(Xtrain[p])
			for p in range(0, Xtest.shape[0]):
				Xtest[p] = sc_X.fit_transform(Xtest[p])
		else:
			Xtrain = sc_X.fit_transform(Xtrain)
			Xtest = sc_X.transform(Xtest)
        
		return Xtrain, ytrain, Xtest, ytest  
	   
	def xy_simple(self):
		#new x_simple: one large array, flatten
		#['test0 test1 test2..'] NO gender
		nr = self.data.shape[0] #pid
		nc = self.data.shape[1] * self.data.shape[2] #test/time
		x_simple = np.zeros((nr, nc))
		y_simple = np.zeros((nr,1))
		i = 0 #thru pid
		for r in range(0,nr):
			print("pid", i)
			j = 0 #thru test
			k = 0 #thru time
			for c in range(0,nc):
				x_simple[r,c] = self.data[i][j][k,2] #r0/t0..r0/t59, r1/t0			
				print("pid/test/time for row/col", i, j, k, "|", r, c)
				if k == (self.data.shape[2]-1): #fill all time per test
					j += 1 #next test
					k = 0 #reset time
				else:                    
					k += 1 #each col = next time same test    
			#pdb.set_trace()			           
			y_simple[r] = self.data[i][0][0,-1] #infection per pid
			i += 1 #each row = next pid
            
		pdb.set_trace()
		print(x_simple.shape, y_simple.shape)		
		Xtrain, ytrain, Xtest, ytest = self.xy_output(x_simple, y_simple, 0)
		return Xtrain, ytrain, Xtest, ytest

	def conv_xy_simple(self):
		#new conv_xy_simple (Npid, tsample, tests)
		nr = self.data.shape[2] #tsample
		nc = self.data.shape[1] #tests
		npid = self.data.shape[0]
		x_simple = np.zeros((npid, nr, nc))
		y_simple = np.zeros((npid,1))
		for p in range(0, npid):
			for c in range(0, nc):
				x_simple[p][:,c] = self.data[p][c][:,2]
			y_simple[p] = self.data[p][0][0,-1]
		
		print(x_simple.shape, y_simple.shape)
		Xtrain, ytrain, Xtest, ytest = self.xy_output(x_simple, y_simple, 1) #1=3D arr
		return Xtrain, ytrain, Xtest, ytest
		
		
		
class impute_preprocess:
	"""
	pre-process for impute algorithms: Forward, PCA
	"""
	def __init__(self, xy, rel2, nSamples, method):
		self.xy = np.array(xy)
		self.nSamples = nSamples
		self.rel2 = rel2 #min
		self.method = method
		
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
					#(typeTests, Nsamples, attributes)
					allTest.append(non_impute)
				else:					
					#pdb.set_trace() 
					if self.method == 'LOCF':                   
						LOCF_impute = self.impute_LOCF_bTest(match, t_window)
						#(typeTests, Nsamples, attributes)
						allTest.append(LOCF_impute)
					elif self.method == 'NN':
						NN_impute = self.impute_NN_bTest(match, t_window)
						allTest.append(NN_impute)
					elif self.method == 'PCA':
						PCA_impute = self.impute_PCA_bTest(match, t_window)	
						allTest.append(PCA_impute)
				
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

	#Impute NN: nearest value
	def impute_NN_bTest(self, arr, t_window):
		arr = np.array(sorted(arr, key=itemgetter(3), reverse=True))
		#print("input arr", arr)
		slist_forward_impute = []
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
			slist_forward_impute.append(stmp)
		#pdb.set_trace()
		return slist_forward_impute
		
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
		#self.x1 = self.x1[0:104675,:]
		
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
			
			#investigate patients that are dropped
			'''patient_drop = np.array(patient_drop)
			patient_drop_check = check_unique(patient_drop[:,0]) #unique pid drop 745
			patient_kept = check_unique(self.xnew[:,0]) #unique pid kept 778
			patient_kept = list(patient_kept)
			patient_drop_real = []						
			for r1, elem1 in enumerate(patient_drop): #loop thru each drop pid
			    if(all(elem1[0] != r2[0] for r2 in patient_kept)):
			        #pid, testDate, testType, numAns, inf, infDate, surgDate, gen, yob, surg_test_date, surg_inf_date
			        #print(elem1)
			        patient_drop_real.append(elem1) 
			patient_drop_real = np.array(patient_drop_real)
			if patient_drop_real.shape[0] != 0:
			    patient_drop_real_check = check_unique(patient_drop_real[:,0]) #51 pid
			    #i.e.: pid: 137765 closest test date 2004 for surgery 2002
			else:
			    print("0 unique pid dropped")'''                  
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




