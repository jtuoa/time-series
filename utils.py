import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from operator import itemgetter
import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pdb
import math

DAY_MIN = 1440
HR_MIN = 60
REL_DAYRANGE = 60 #day prior knowledge
NORMC = 2.0

def check_unique(arr):
	u,c = np.unique(arr, return_counts=True)
	uc_table = np.asarray((u,c)).T
	print(uc_table)
	print("uc_table shape: ", uc_table.shape)
	return uc_table


class more_preprocess:
	"""
	pre-process by embed blood test
	"""
	def __init__(self, xy, testName):
		self.xy = xy
		self.testName = testName
	
	def get_test_idx(self):
		for i in range(0, self.testName.shape[0]):
			self.xy[:,i,:,3] = i

	def normalize_test(self):
		for i in range(0, self.xy.shape[0]):
			for j in range(0, self.testName.shape[0]):
				if (max(self.xy[i,j,:,4]) == 0):
					self.xy[i,j,:,4] = self.xy[i,j,:,4]
				else:
				    self.xy[i,j,:,4] = self.xy[i,j,:,4] / max(self.xy[i,j,:,4]) #subtract mean / new max [-1,1]
		return self.xy
		#self.test_result = self.xy[:,:,:,3:5] #[testName, normTest]
		

        	
class snapShot_preprocess:
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

class impute_preprocess:
	"""
	pre-process for impute algorithms: Forward, PCA
	"""
	def __init__(self, xy, nSamples):
		self.xy = np.array(xy)
		self.nSamples = nSamples
		
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
			#subarray base on test
			for r2, elem2 in enumerate(t_uTest):	
				match = [x for x in list(self.xy) if x[3] == elem2 and x[0] == elem1]
				match_arr = np.array(sorted(match, key=itemgetter(1), reverse=True))
				
				#pdb.set_trace()
				#nothing to interpolate
				if match_arr.size == 0: #enter (4, 3, 20, 8)
					mean = self.mean_perTest(elem2)
					non_impute = self.create_nonImpute_list(elem1, r1, elem2, t_window, mean)
					#(t_uTest size, samples, col)
					allTest.append(non_impute)
				else:					
					#IMPUTE METHOD CHANGE HERE!!!
					forward_impute = self.impute_forward_bTest(match_arr, t_uTest, t_window)		
					#(t_uTest size, samples, col)
					allTest.append(forward_impute)
				
			#pdb.set_trace()
			#(PID, t_uTest size, samples, col)
			PID_allTest.append(allTest)
			check = np.array(PID_allTest)
			print(check.shape)
		#pdb.set_trace() 
		return PID_allTest
	
	#impute forward
	def impute_forward_bTest(self, arr, t_uTest, t_window):		
		#compute mean across all patients for test
		#find number closest to sample point
		slist_forward_impute = []
		for tval in t_window:
			realClosestTime = 1000e9 #define very large # to update
			for arow, aval in enumerate(arr[:,1]):
				closestTimeCheck = abs(tval - aval)
				if closestTimeCheck < realClosestTime:
					realClosestTime = closestTimeCheck
					realIdx = arow

			#create new sample matrix
			stmp = list(arr[realIdx,:])
			stmp[1] = tval
			#pdb.set_trace()
			slist_forward_impute.append(stmp)
			#print("closest ", realClosestTime, realIdx)		
		return slist_forward_impute
	
	def mean_perTest(self, test):
		#compute mean across all patients for test
		match = [x for x in list(self.xy) if x[3] == test]
		match_arr = np.array(match)
		mean = np.sum(match_arr[:,4])/match_arr.shape[0]
		#mean_list = [mean]*self.nSamples
		return mean
			
	#for PID that dont have any results in the test
	def create_nonImpute_list(self, pid, pidIdx, test, t_window, mean):
		#create one row
		arr = np.empty([self.nSamples, 16], dtype=object)
		arr[:,0].fill(pid)
		arr[:,1] = t_window
		arr[:,2] = abs(1/arr[:,1]/NORMC)
		arr[:,3].fill(test)
		arr[:,4].fill(mean)
		arr[:,5].fill(self.xy[0][5]) #gender
		arr[:,6].fill(self.xy[0][6]) #Yob integer encode 1
		arr[:,7].fill(self.xy[0][7]) #Yob integer encode 2
		arr[:,8].fill(self.xy[0][8]) #Yob integer encode 3
		arr[:,9].fill(self.xy[0][9]) #Yob integer encode 4
		arr[:,10].fill(self.xy[0][10]) #Yob integer encode 5
		arr[:,11].fill(self.xy[0][11]) #Yob integer encode 6
		arr[:,12].fill(self.xy[0][12]) #Yob integer encode 7
		arr[:,13].fill(self.xy[0][13]) #Yob integer encode 8
		arr[:,14].fill(self.xy[0][14]) #Yob integer encode 9
		arr[:,15].fill(self.xy[0][15])#infection
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
		#x1 = x1[0:10000,:] #for test
		
		dataset2 = pd.read_csv(self.filename2, delimiter='\t', encoding='utf-8')
		dataset2.fillna(0, inplace = True)
		self.x2 = dataset2.loc[:,['PID', 'Infection', 't.IndexSurgery', 'Sex', 'YoB']]
		self.x2 = np.array(self.x2)
	
	def keep_ntopTests(self):
		#top unique test elements: common blood tests more meaningful
		#use ntopTests = top blood test results
		u, c = np.unique(self.x1[:,2], return_counts=True)
		uc_table = np.asarray((u,c)).T
		testType_sort = sorted(uc_table, key=itemgetter(1), reverse=True)
		ntopTests_type = testType_sort[0:self.ntopTests]
		#x1 keep only top tests by append to x1new
		self.x1new = []
		for r1, elem1 in enumerate(self.x1):
			if(any(elem1[2] == r2[0] for r2 in ntopTests_type)):
				self.x1new.append(elem1)
		self.x1new = np.array(self.x1new)
		#pdb.set_trace()
	
	def combine_datasets(self):
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
		normc = min(abs(self.xnew[:,8])) #norm constant by newDate
		print("NORM CONSTANT", normc)
		test_surgery_time_weight = abs(1/(self.xnew[:,8]/normc)) #range[0,1]
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
		yob_compare = (np.array((self.xnew[:,7], np.array(yob_group)))).T.astype(int)
		#pdb.set_trace()
		#check_uTable = check_unique(yob_compare[:,1])
		#print(check_uTable)
		#yob_integer_encoded = np.array(yob_group)
		#self.xnew = np.concatenate((self.xnew, yob_integer_encoded.reshape(yob_integer_encoded.shape[0],1)), axis=1)

		#convert yob_group, integer format to one hot encoder
		yob_onehot_encoded = to_categorical(yob_group)
		self.xnew = np.concatenate((self.xnew, yob_onehot_encoded), axis=1) #yob tag after
	
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
		dfxy_all = pd.DataFrame(self.xnew, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.IndexSurgery', 'Sex', 'YoB', 'newDate', 'newDate_weight', 'Gender_encode', 'Yob_intencode1', 'Yob_intencode2', 'Yob_intencode3', 'Yob_intencode4', 'Yob_intencode5', 'Yob_intencode6', 'Yob_intencode7', 'Yob_intencode8', 'Yob_intencode9'])
		
		dfxy = dfxy_all.loc[:,['PID', 'newDate', 'newDate_weight', 'TestType', 'NumAnswer', 'Gender_encode', 'Yob_intencode1', 'Yob_intencode2', 'Yob_intencode3', 'Yob_intencode4', 'Yob_intencode5', 'Yob_intencode6', 'Yob_intencode7', 'Yob_intencode8', 'Yob_intencode9', 'Infection']]
		
		'''dfxy_all = pd.DataFrame(self.xnew, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.IndexSurgery', 'Sex', 'YoB', 'newDate', 'newDate_weight', 'Gender_encode', 'Yob_int'])
		
		dfxy = dfxy_all.loc[:,['PID', 'newDate', 'newDate_weight', 'TestType', 'NumAnswer', 'Gender_encode', 'Yob_intencode', 'Infection']]'''
		return dfxy, testName




