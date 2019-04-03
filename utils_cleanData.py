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
REL_DAYRANGE = 0 #60 #day prior knowledge 60, POSTOPERATIVE = 0


def check_unique(arr):
	u,c = np.unique(arr, return_counts=True)
	uc_table = np.asarray((u,c)).T
	print(uc_table)
	print("uc_table shape: ", uc_table.shape)
	return uc_table


	
class data_preprocess:
	def __init__(self, filename1, filename2, ntopTests, nYob, ntestRel_useRange, numKeepTests=5):
		self.filename1 = filename1
		self.filename2 = filename2
		self.ntopTests = ntopTests
		self.nYob = nYob
		self.ntestRel_useRange = ntestRel_useRange
		self.numKeepTests = numKeepTests
		
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
	
	def keep_oneTest(self):
		testName = 'ALP'
		self.x1new = [x for x in list(self.x1new) if x[2] == testName]
		self.x1new = np.array(self.x1new)
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
		#self.rel2 = 0 #PREOPERATIVE
		#pdb.set_trace()
	
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
	
	def discard_tests_pid(self):
		#ensure all pid has at least 5 results per test
		p = check_unique(self.xnew[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		t = check_unique(self.xnew[:,2]) #testType
		t_arr = np.array(t)
		t_uTest = t_arr[:,0]
		
		#drop tests with less than 5 values
		#pdb.set_trace()
		self.xnew = list(self.xnew)
		for r1, elem1 in enumerate(p_uTest):
			for r2, elem2 in enumerate(t_uTest):
				match_indices = [i for i, x in enumerate(self.xnew) if x[0] == elem1 and x[2] == elem2]
				if len(match_indices) < self.numKeepTests:
					for match_indices in sorted(match_indices, reverse=True):
						del self.xnew[match_indices]
						
					check_del = [x for x in self.xnew if x[0] == elem1 and x[2] == elem2]
					assert len(check_del) == 0, "Error: incorrect test delete for pid"
			print("discard_tests_pid patient:", r1)
		
		#if still any pid has no results at the blood test drop pid completely
		#pdb.set_trace()
		self.xnew = np.array(self.xnew)
		p = check_unique(self.xnew[:,0]) #pid
		p_arr = np.array(p)
		p_uTest = p_arr[:,0]
		
		for r1, elem1 in enumerate(p_uTest):
			for r2, elem2, in enumerate(t_uTest):
				match_indices = [i for i, x in enumerate(list(self.xnew)) if x[0] == elem1 and x[2] == elem2]
				if len(match_indices) == 0:
					print("dropping pid", elem1)
					self.xnew = self.xnew[self.xnew[:,0] != elem1]
					break
			check_del = [x for x in list(self.xnew) if x[0] == elem1]
			assert len(check_del) == 0, "Error: incorrect test delete for pid"
						            
		p = check_unique(self.xnew[:,0])				
		pdb.set_trace()
		
	
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
