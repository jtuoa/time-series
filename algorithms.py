import numpy as np
import pandas as pd
import pdb
from utils import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Activation, Input, Embedding, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

'''import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #dissable warnings and debug info from TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #dynamically grow the memory used on GPU
sess = tf.Session(config=config)
K.set_session(sess) #set this tf session as default session for keras'''


class MLP:
	"""
	Multilayer preceptron
	"""
	def __init__(self):
		self.params={'nbatch':32, 'nepoch':20, 'nTest':50, 'nSamples':20}
		
	def train(self, xtrain, ytrain, xvalid, yvalid):
		
		test_input = Input(shape=(xtrain.shape[0], self.params['nTest'], self.params['nSamples'], )) #test_type
		test_auxin = Input(shape=(xtrain.shape[0], self.params['nTest'], self.params['nSamples'], 3)) #date, weight, result
		attr_input = Input(shape=(xtrain.shape[0], self.params['nTest'], self.params['nSamples'], 10)) #gender, yob
				
		x = Embedding(input_dim=self.params['nTest'], output_dim=32)(test_input)
		x = concatenate([x, test_auxin, attr_input])
		
		x = Dense(512, activation='relu')(x) #num layers
		x = Dense(512, activation='relu')(x)
		x = Dense(512, activation='relu')(x)		
		has_infection = Dense(1, activation="sigmoid")(x)		
		self.model = Model(inputs=[test_input, test_auxin, attr_input], outputs=has_infection)
		
		self.model.compile(optimizer='adam', loss="binary_crossentropy")
		test_inputX = xtrain[:,:,:,3]
		test_auxinX = xtrain[:,:,:,[1,2,4]]
		attr_inputX = xtrain[:,:,:,5:15]
		#can't use x=xtrain error: expect to see 3 arrays
		#can't use test_inputX=xtrain[:,:,:,3:4] err: expect(708,50,20) got (50,20,1)
		#x=[] err: input_q to have 4 dim got (708,50,20)		
		self.model.fit(x=[test_inputX, test_auxinX, attr_inputX], y=ytrain, epochs=self.params['nepoch'], batch_size=self.params['nbatch'], verbose=0, validation_data=(xvalid, yvalid))
		
		#self.model.save("/home/juiwen/Documents/CMPUT659/mp1.h5")
		#print(self.model.summary())
		#plot_model(self.model, to_file='multilayer_perceptron_graph.png')
	
	def grid_search(self, X, Y):
		#grid search parameters
		batch_size = [16, 32, 64]
		epochs = [10, 20, 30]
		optimizer = ['SGD', 'RMSprop', 'Adam']
		param_grid = dict(batch_size = batch_size, epochs = epochs, optimizer = optimizer)
		grid = GridSearchCV(estimator = self.model, param_grid = param_grid, n_jobs=-1)
		grid_result = grid.fit(X,Y)
		print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, std, param in zip(means, stds, params):
			print("%f(%f) with : %r" % (mean, stdev, param))
	
	def predict(self, xtest):
		ytest = self.model.predict(xtest)
		return ytest

class LR:
	"""
	Logistic regression
	"""
	def __init__(self):
		self.params={'nbatch':32, 'nepoch':20, 'nTest':50}
		
	def train(self, xtrain, ytrain, xvalid, yvalid):
		
		test_input = Input(shape=(self.params['nTest'],)) #Input1: test_type
		test_auxin = Input(shape=(self.params['nTest'],3)) #Input2: date, weight, result
		attr_input = Input(shape=(2,)) #Input3: gender, yob
		
		x = Embedding(input_dim=self.params['nTest'], output_dim=32)(test_input)
		x = concatenate([x, test_auxin, attr_input]) #merge input models
		
		has_infection = Dense(1, activation="sigmoid")(x)		
		self.model = Model(input=[test_input, test_auxin, attr_input], output=has_infection)
		
		self.model.compile(optimizer="adam", loss="binary_crossentropy")		
		self.model.fit(x=xtrain, y=ytrain, epochs=self.params['nepoch'], batch_size=self.params['nbatch'], verbose=0, validation_data=(xvalid, yvalid))
		
		#self.model.save("/home/juiwen/Documents/CMPUT659/mp1.h5")
		#print(self.model.summary())
		#plot_model(self.model, to_file='multilayer_perceptron_graph.png')
		
	
	def predict(self, xtest):
		ytest = self.model.predict(xtest)
		return ytest


class snapShot:
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

