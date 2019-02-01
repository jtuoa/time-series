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
	def snapshot_plot(self,xy_3D):
		#color code search: plotting different colors in matplotlib 
		c = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 
		m = ['*', 'o', 'v', '^', '<', '1', 's', 'x', 'd']
		#ls = ['-', ':'] # - = no infection
		for i in range(0, len(xy_3D)):
			cidx = 0
			midx = 0
			#print("Table of unique PID per test")
			#cu = check_unique(xy_3D[i][:,0])
			#print(cu)
			#print("Infection :", list(xy_3D[0][:,0]).index(cu[0]))
			const = abs(min(xy_3D[0][:,3])) + 1e-6
			for j in range(0, len(xy_3D[0])-1):
				print(j)
				#plot same color for same PID
				if (xy_3D[i][j][0] - xy_3D[i][j+1][0]) == 0:
					#x = time, y = number
					'''if xy_3D[i][j][4] == 0: 
						lsidx = 0
					else:
						lsidx = 1'''
					plt.plot(((xy_3D[i][j][3])/max(xy_3D[0][:,3])), xy_3D[i][j][2], color=c[cidx], marker=m[midx])
					plt.savefig('myfig'+str(i))
				else:
					pdb.set_trace()
					plt.plot(((xy_3D[i][j][3])/max(xy_3D[0][:,3])), xy_3D[i][j][2], color=c[cidx], marker=m[midx])
					plt.savefig('myfig'+str(i))
					if midx == len(m)-1:
						cidx += 1
						midx = 0
					midx += 1
			
			#for last row
			if xy_3D[i][len(xy_3D[0])-1][0] == xy_3D[i][len(xy_3D[0])-2][0]:
				plt.plot(xy_3D[i][len(xy_3D[0])-1][3], xy_3D[i][len(xy_3D[0])-1][2], color=c[cidx], marker=m[midx])
				plt.savefig('myfig'+str(i))
			else:
				if midx == len(m)-1:
					cidx += 1
					midx = 0
				midx += 1
				plt.plot(xy_3D[i][len(xy_3D[0])-1][3], xy_3D[i][len(xy_3D[0])-1][2], color=c[cidx], marker=m[midx])
				plt.savefig('myfig'+str(i))

#impute data prep
def impute_data(x1new):
	x1new = np.array(x1new)
	p = check_unique(x1new[:,0])
	p_arr = np.array(p)
	p_uTest = p_arr[:,0]
	
	t = check_unique(x1new[:,3])
	t_arr = np.array(t)
	t_uTest = t_arr[:,0]
	
	PID_allTest = []	
	#for each PID
	for r1, elem1 in enumerate(p_uTest):
		allTest = []
		#subarray base on test
		for r2, elem2 in enumerate(t_uTest):	
			match = [x for x in list(x1new) if x[3] == elem2 and x[0] == elem1]
			match_arr = np.array(sorted(match, key=itemgetter(1), reverse=True))
			
			pdb.set_trace()
			#nothing to interpolate
			if match_arr.size == 0:
				break
			#impute per PID per test
			forward_impute = impute_forward_bTest(match_arr, t_uTest)
			
			#(t_uTest size, samples, col)
			allTest.append(forward_impute)
		pdb.set_trace()
		#(PID, t_uTest size, samples, col)
		PID_allTest.append(allTest)
		check = np.array(PID_allTest)
		print(check.shape)
	pdb.set_trace() 
	return PID_allTest
	
#impute forward
def impute_forward_bTest(arr, t_uTest):
	nSamples = 20
	t_window = np.linspace(-REL_DAYRANGE*DAY_MIN, REL_DAYRANGE*DAY_MIN, nSamples)
	t_window = sorted(t_window, reverse=True)
	
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
	
	#compute mean across all patients for test
	'''for r1, elem1 in enumerate(t_uTest):		
		match = [x for x in list(x1new) if x[3] == elem1]
		match_arr = np.array(match)
		mean = np.sum(match_arr[:,4])/match_.shape[0]
		mean_arr[r1,1] = mean'''
	return slist_forward_impute
		
def check_unique(arr):
	u,c = np.unique(arr, return_counts=True)
	uc_table = np.asarray((u,c)).T
	print(uc_table)
	print("uc_table shape: ", uc_table.shape)
	return uc_table

def keep_ntopTests(x1, ntopTests):
	#top unique test elements: common blood tests more meaningful
	#use ntopTests = top blood test results
	u, c = np.unique(x1[:,2], return_counts=True)
	uc_table = np.asarray((u,c)).T
	testType_sort = sorted(uc_table, key=itemgetter(1), reverse=True)
	ntopTests_type = testType_sort[0:ntopTests]
	#x1 keep only top tests by append to x1new
	x1new = []
	for r1, elem1 in enumerate(x1):
		if(any(elem1[2] == r2[0] for r2 in ntopTests_type)):
			x1new.append(elem1)	
	return x1new

def create_newDateCol(x1new):
	#create newDate col (rel minutes) = surgery date - test date
	x1new = np.hstack((x1new, np.zeros((x1new.shape[0], 1), dtype=x1new.dtype)))
	for i in range(0, len(x1new)):
		surgery_time = (datetime.datetime.strptime(x1new[i,5], '%Y-%m-%d %H:%M:%S')).strftime('%d/%m/%y %H:%M')
		test_surgery_time = datetime.datetime.strptime(surgery_time, '%d/%m/%y %H:%M') - datetime.datetime.strptime(x1new[i,1], '%d/%m/%y %H:%M')
		#store relative minutes
		if test_surgery_time.days < 0:
			test_surgery_time_sec = -1 * test_surgery_time.seconds
		else:
			test_surgery_time_sec = +1 * test_surgery_time.seconds
		#pos = test before surgery, neg = test after surgery
		x1new[i,x1new.shape[1]-1] = (test_surgery_time.days*DAY_MIN) + test_surgery_time_sec/HR_MIN
	return x1new

def discard_testRelRange(x1new):
	x2new = []
	rel_minRange = REL_DAYRANGE * DAY_MIN
	for r1, elem1 in enumerate(x1new):
		if(elem1[8] <= rel_minRange and elem1[8] >= -rel_minRange): #time window
			x2new.append(elem1)
	x2new=np.array(x2new)
	return x2new

def create_weightVec_newDateCol(x1new):
	#newDate weight_vector
	#heavier weight when x1new[:,8] smaller
	normc = min(abs(x1new[:,8])) #norm constant
	test_surgery_time_weight = abs(1/(x1new[:,8]/normc)) #range[0,1]
	x1new = np.hstack((x1new, test_surgery_time_weight.reshape(len(test_surgery_time_weight),1)))
	return x1new

def create_yobGroup(x1new, nYob):
	min_yob = min(x1new[:,7])
	max_yob = max(x1new[:,7])
	yob_range = np.arange(min_yob, max_yob+nYob, nYob, dtype=int)
	#find values in range and replace with integer (group)
	yob_group = list(x1new[:,7])
	for i in range(0, len(yob_range)-1):
		yob_group = list(map(lambda x: i if (x>=yob_range[i]) and (x<yob_range[i+1]) else x, yob_group))
	yob_group[:] = [len(yob_range)-2 if x == max_yob else x for x in yob_group]
	yob_compare = (np.array((x1new[:,7], np.array(yob_group)))).T.astype(int)
	#check_uTable = check_unique(yob_compare[:,1])
	#print(check_uTable)
	yob_integer_encoded = np.array(yob_group)
	x1new = np.concatenate((x1new, yob_integer_encoded.reshape(yob_integer_encoded.shape[0],1)), axis=1)

	#convert yob_group, integer format to one hot encoder
	#yob_onehot_encoded = to_categorical(yob_group)
	#x1new = np.concatenate((x1new, yob_onehot_encoded), axis=1)
	return x1new		

'''def embed_bloodTest(x1new):
	doc = x1new[:,2]
	doc = list(doc)
	#labels = [
	
	#integer encode the documents
	doc_size = len(doc)
	encoded_docs_oe = [one_hot(d,doc_size) for d in doc]
	
	#definemodel
	model = Sequential()
	model.add(Embedding(input_dim = doc_size, output_dim=32, input_length=1))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	
	return x1new #EDIT!'''
		

def load_data(filename1, filename2, ntopTests, nYob, ntestRel_useRange):
	#load both datasets
	dataset1 = pd.read_csv(filename1, delimiter='\t', encoding='utf-8')
	dataset1.fillna(0, inplace = True)
	x1 = dataset1.loc[:,['PID', 'Date', 'TestType', 'NumAnswer']]
	x1 = np.array(x1)
	#x1 = x1[0:10000,:] #for test
	
	dataset2 = pd.read_csv(filename2, delimiter='\t', encoding='utf-8')
	dataset2.fillna(0, inplace = True)
	x2 = dataset2.loc[:,['PID', 'Infection', 't.IndexSurgery', 'Sex', 'YoB']]
	x2 = np.array(x2)
	
	#keep only top blood test rows
	x1new = keep_ntopTests(x1, ntopTests)
	x1new = np.array(x1new)
	
	#combine train & label datasets
	x1new = np.hstack((x1new, np.zeros((x1new.shape[0], 4), dtype=x1new.dtype)))
	for r1, elem1 in enumerate(x1new):
		r2 = list(x2[:,0]).index(elem1[0])
		x1new[r1,4:x1new.shape[1]] = x2[r2,1:x2.shape[1]]	
	del r1, elem1, r2
	
	#delete rows with infection=2
	x1new = pd.DataFrame(x1new, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.IndexSurgery', 'Sex', 'YoB'])
	x1new=x1new[x1new.Infection != 2]
	x1new = np.array(x1new)
	
	#create newDateCol = surgery - test date
	x1new = create_newDateCol(x1new)
	
	#choose to discard tests with dates outside ntestRel_dayRange
	if ntestRel_useRange:
		x1new = discard_testRelRange(x1new)
	
	#create weight vector for newDateCol
	x1new = create_weightVec_newDateCol(x1new)
	
	#encode gender
	label_encoder = LabelEncoder()
	gender_integer_encoded = label_encoder.fit_transform(x1new[:,6])
	x1new = np.hstack((x1new, gender_integer_encoded.reshape(len(gender_integer_encoded),1)))
	
	#pdb.set_trace()
	#encode YoB by user specify nYob
	x1new = create_yobGroup(x1new, nYob)
	
	#embed blood test
	#x1new = embed_bloodTest(x1new)

	#pdb.set_trace()
	table = check_unique(x1new[:,0])
	
	#new dataset
	dfxy_all = pd.DataFrame(x1new, columns=['PID', 'Date', 'TestType', 'NumAnswer', 'Infection', 't.IndexSurgery', 'Sex', 'YoB', 'newDate', 'newDate_weight', 'Gender_encode', 'Yob_intencode'])
	
	dfxy = dfxy_all.loc[:,['PID', 'newDate', 'newDate_weight', 'TestType', 'NumAnswer', 'Gender_encode', 'Yob_intencode', 'Infection']]

	return dfxy


