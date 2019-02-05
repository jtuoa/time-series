import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from operator import itemgetter
import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
#from hypertools.hypertools.tools.reduce import reduce as reducer
from ppca.ppca import PPCA
import copy
import pdb

DAY_MIN = 1440
HR_MIN = 60
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
	def __init__(self, xy, normc, nSamples):
		self.xy = np.array(xy)
		self.nSamples = nSamples
		self.normc = normc
		
	#impute data prep
	def impute_data(self):
		p = check_unique(self.xy[:,0])
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		t = check_unique(self.xy[:,3])
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
		
		t_window = np.linspace(-REL_DAYRANGE*DAY_MIN, REL_DAYRANGE*DAY_MIN, self.nSamples)
		t_window = sorted(t_window, reverse=True)
		
		PID_allTest = []	
		#for each PID
		for r1, elem1 in enumerate(p_uTest):
			allTest = []
			#search for all the time testA is done for this pid = match
			for r2, elem2 in enumerate(t_uTest):	
				match = [x for x in list(self.xy) if x[3] == elem2 and x[0] == elem1]
				match_arr = np.array(sorted(match, key=itemgetter(1), reverse=True))
				
				#pdb.set_trace()
				#Empty: there is no testA for this pid
				if match_arr.size == 0:
					#pdb.set_trace()
					mean = self.mean_perTest(elem2) #find mean of across all pid
					xy_col = self.xy.shape[1]
					non_impute = self.create_nonImpute_list(elem1, elem2, t_window, mean, xy_col)
					#(t_uTest size, samples, col)
					allTest.append(non_impute)
				else:					
					#IMPUTE METHOD CHANGE HERE!!!
					#pdb.set_trace()                    
					forward_impute = self.impute_forward_bTest(match_arr, t_window)
					#pca_impute = self.impute_PCA_bTest(match_arr, t_window)		
					#(t_uTest size, samples, col)
					allTest.append(forward_impute)
				
			#pdb.set_trace()
			#(PID, t_uTest size, samples, col)
			PID_allTest.append(allTest)
			check = np.array(PID_allTest)
			print(check.shape)
		#pdb.set_trace() 
		return PID_allTest
	
	#impute mean: take avg across the time frame if none then use default
	
	#impute PCA
	def impute_PCA_bTest(self, arr, t_window):
		old_arr = copy.deepcopy(arr)
		#create nan list
		slist_pca_impute = []
		for i in range(len(t_window)):
			arr[0][1] = t_window[i] #time in col 1			
			arr[0][4] = np.nan #result in col 4
			slist_pca_impute.append(list(arr[0]))
		pdb.set_trace()
		#find number closest to sample point and fill ONCE		
		for arow, aval in enumerate(old_arr[:,1]):
			minVal = [x - aval for x in t_window]
			minIdx = np.argmin(abs(np.array(minVal)))
			stmp = list(old_arr[arow,:])
			stmp[1] = t_window[minIdx]
			#insert stmp into slist at minIdx
			slist_pca_impute[minIdx] = stmp
		
		#implement PCA on nan
		#if only 0 and nan in col 4 then can't PCA
		'''sarr_pca_impute = np.array(slist_pca_impute)
		det = sarr_pca_impute[:,4]
		inds = [i for i,n in enumerate(det) if str(n) == "nan" or str(n) == "0.0"]
		if len(inds) != len(slist_pca_impute): #can PCA
			data_r = hyp.reduce(sarr_pca_impute, ndims=3)'''
		#just PCA regardless of only 0 and nan in col 4
		#data_r = reducer(sarr_pca_impute, ndims=3)
		pca = PPCA()
		pca.fit(sarr_pca_impute, 3)
		
		return list(data_r) #row	

	#Impute forward: find time closest to sample point
	def impute_forward_bTest(self, arr, t_window):		
		slist_forward_impute = []
		for tval in t_window:
			realClosestTime = 100e9
			for arow, aval in enumerate(arr[:,1]):
				closestTimeCheck = abs(tval - aval)
				if closestTimeCheck < realClosestTime:
					realClosestTime = closestTimeCheck
					realIdx = arow
			stmp = list(arr[realIdx,:])
			stmp[1] = tval
			slist_forward_impute.append(stmp)
		return slist_forward_impute
	
    #Fill by: mean over all pid for pid with no measurements in testA
	def mean_perTest(self, test):
		#compute mean across all patients for test
		match = [x for x in list(self.xy) if x[3] == test]
		match_arr = np.array(match)
		mean = np.sum(match_arr[:,4])/match_arr.shape[0]
		return mean
			
	#Create list: for PID that dont have any measurement in testA
    #[pid, time, time_weight, test, numAnswer, gender, 9yob, infection]
	def create_nonImpute_list(self, pid, test, t_window, mean, xy_col):
		arr = np.empty([self.nSamples, xy_col], dtype=object)
		arr[:,0].fill(pid)
		arr[:,1] = t_window
		arr[:,2] = abs(1/arr[:,1]/self.normc)
		arr[:,3].fill(test)
		arr[:,4].fill(mean)
		for i in range(5, xy_col):
			arr[:,i].fill(self.xy[0][i]) #gender, 9yob, infection
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
		self.x1 = np.array(self.x1)
		
		dataset2 = pd.read_csv(self.filename2, delimiter='\t', encoding='utf-8')
		dataset2.fillna(0, inplace = True)
		self.x2 = dataset2.loc[:,['PID', 'Infection', 't.IndexSurgery', 'Sex', 'YoB']]
		self.x2 = np.array(self.x2)
	
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
		self.x1new = np.array(self.x1new)
		#pdb.set_trace()
	
	def combine_datasets(self):
        #combine x1 measurement, x2 metadata
		self.x1new = np.hstack((self.x1new, np.zeros((self.x1new.shape[0], 4), dtype=self.x1new.dtype)))
		for r1, elem1 in enumerate(self.x1new):
			r2 = list(self.x2[:,0]).index(elem1[0])
			self.x1new[r1,4:self.x1new.shape[1]] = self.x2[r2,1:self.x2.shape[1]]	
		#pdb.set_trace()
		
	def delete_infection2(self):
		self.x1new = pd.DataFrame(self.x1new, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.IndexSurgery', 'Sex', 'YoB'])
		self.x1new = self.x1new[self.x1new.Infection != 2]
		self.x1new = np.array(self.x1new)
		#pdb.set_trace()
	
	def create_newDataCol(self):
		#create newDate col (rel minutes) = surgery date - test date
		self.x1new = np.hstack((self.x1new, np.zeros((self.x1new.shape[0], 1), dtype=self.x1new.dtype)))
		for i in range(0, len(self.x1new)):
			surgery_time = (datetime.datetime.strptime(self.x1new[i,5], '%Y-%m-%d %H:%M:%S')).strftime('%d/%m/%y %H:%M')
			test_surgery_time = datetime.datetime.strptime(surgery_time, '%d/%m/%y %H:%M') - datetime.datetime.strptime(self.x1new[i,1], '%d/%m/%y %H:%M')
			#store relative minutes
			if test_surgery_time.days < 0:
				test_surgery_time_sec = -1 * test_surgery_time.seconds
			else:
				test_surgery_time_sec = +1 * test_surgery_time.seconds
			#pos = test before surgery, neg = test after surgery
			self.x1new[i,self.x1new.shape[1]-1] = (test_surgery_time.days*DAY_MIN) + test_surgery_time_sec/HR_MIN
		#pdb.set_trace()
	
	def discard_testRelRange(self):
		if self.ntestRel_useRange:
			self.xnew = []
			rel_minRange = REL_DAYRANGE * DAY_MIN
			for r1, elem1 in enumerate(self.x1new):
				if(elem1[8] <= rel_minRange and elem1[8] >= -rel_minRange): #time window
					self.xnew.append(elem1)
			self.xnew=np.array(self.xnew)
		#pdb.set_trace()
	
	def create_weightVec_newDataCol(self):
		#newDate weight_vector
		#heavier weight when x1new[:,8] smaller
		self.normc = min(abs(self.xnew[:,8])) #norm constant by newDate
		test_surgery_time_weight = abs(1/(self.xnew[:,8]/self.normc)) #range[0,1]
		self.xnew = np.hstack((self.xnew, test_surgery_time_weight.reshape(len(test_surgery_time_weight),1)))
		#pdb.set_trace()

	def encode_gender(self):
		label_encoder = LabelEncoder()
		gender_integer_encoded = label_encoder.fit_transform(self.xnew[:,6])
		self.xnew = np.hstack((self.xnew, gender_integer_encoded.reshape(len(gender_integer_encoded),1)))
		#pdb.set_trace()
		
	def create_yobGroup(self):
		min_yob = min(self.xnew[:,7])
		max_yob = max(self.xnew[:,7])
		yob_range = np.arange(min_yob, max_yob+self.nYob, self.nYob, dtype=int)
		#find values in range and replace with integer (group)
		yob_group = list(self.xnew[:,7])
		for i in range(0, len(yob_range)-1):
			yob_group = list(map(lambda x: i if (x>=yob_range[i]) and (x<yob_range[i+1]) else x, yob_group))
		yob_group[:] = [len(yob_range)-2 if x == max_yob else x for x in yob_group]
		#yob_compare = (np.array((self.xnew[:,7], np.array(yob_group)))).T.astype(int) #checking
		#pdb.set_trace()
		#check_uTable = check_unique(yob_compare[:,1])
		#print(check_uTable)
		#yob_integer_encoded = np.array(yob_group)
		#self.xnew = np.concatenate((self.xnew, yob_integer_encoded.reshape(yob_integer_encoded.shape[0],1)), axis=1)

		#convert yob_group, integer format to one hot encoder
		yob_onehot_encoded = to_categorical(yob_group)
		self.xnew = np.concatenate((self.xnew, yob_onehot_encoded), axis=1) #yob tag after
	
	def integer_encode_testName(self):
		u, c = np.unique(self.xnew[:,2], return_counts=True) #get unique
		uc_table = np.asarray((u,c)).T
		self.xnew = np.array(sorted(self.xnew, key=itemgetter(2)))
		idx1 = 0
		for i in range(0, len(uc_table)):
			idx2 = uc_table[i,1] + idx1
			self.xnew[idx1:idx2,2] = i
			idx1 = idx2
		return self.xnew
    
	def normalize_test(self):
		u, c = np.unique(self.xnew[:,2], return_counts=True) #get unique
		uc_table = np.asarray((u,c)).T
		idx1 = 0
		for i in range(0, len(uc_table)):
			idx2 = uc_table[i,1] + idx1
			sub_xnew = self.xnew[idx1:idx2,3]					
			if (max(sub_xnew) == 0):
				idx1 = idx2	      
			else: #norm to range [-1, 1]
				sub_xnew_norm = (sub_xnew - np.mean(sub_xnew))/max(sub_xnew)
				self.xnew[idx1:idx2,3] = sub_xnew_norm
				idx1 = idx2	
		return self.xnew    

	def result(self):
		#pdb.set_trace()
		check = check_unique(self.xnew[:,0])
		print(check)
		
		testName = check_unique(self.xnew[:,2])
		testName = testName[:,0]
		
		'''yearUnique = check_unique(self.xnew[:,11])
		print("Year Unique: ", yearUnique)
		yearUnique = yearUnique[:,0]'''
		
		#new dataset
		dfxy_all = pd.DataFrame(self.xnew, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', \
                                              't.IndexSurgery', 'Sex', 'YoB', 'newDate', 'newDate_weight', \
                                              'Gender_encode', 'Yob_intencode1', 'Yob_intencode2', \
                                              'Yob_intencode3', 'Yob_intencode4', 'Yob_intencode5', \
                                              'Yob_intencode6', 'Yob_intencode7', 'Yob_intencode8', \
                                              'Yob_intencode9'])
		
		dfxy = dfxy_all.loc[:,['PID', 'newDate', 'newDate_weight', 'TestType', 'NumAnswer', 'Gender_encode', \
                         'Yob_intencode1', 'Yob_intencode2', 'Yob_intencode3', 'Yob_intencode4', \
                         'Yob_intencode5', 'Yob_intencode6', 'Yob_intencode7', 'Yob_intencode8', \
                         'Yob_intencode9', 'Infection']]
		
        #yob integer encode only
		'''dfxy_all = pd.DataFrame(self.xnew, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.IndexSurgery', 'Sex', 'YoB', 'newDate', 'newDate_weight', 'Gender_encode', 'Yob_int'])
		
		dfxy = dfxy_all.loc[:,['PID', 'newDate', 'newDate_weight', 'TestType', 'NumAnswer', 'Gender_encode', 'Yob_intencode', 'Infection']]'''
		return dfxy, self.normc




