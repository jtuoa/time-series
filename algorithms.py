import numpy as np
import pandas as pd
import pdb
from utils import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Activation, Input
import tensorflow as tf

'''import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #dissable warnings and debug info from TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #dynamically grow the memory used on GPU
sess = tf.Session(config=config)
K.set_session(sess) #set this tf session as default session for keras'''


class MP:
	"""
	Multilayer preceptron
	"""
	def __init__(self):
		self.params={'nbatch':32, 'nepoch':20, 'nTest':50}
		
	def train(self, xtrain, ytrain, xvalid, yvalid):
		
		test_input = Input(shape=(self.params['nTest'],)) #test_type
		test_auxin = Input(shape=(self.params['nTest'],3)) #date, date_weight, test_result
		attr_input = Input(shape=(2,)) #gender, yob
		
		x = Embedding(input_dim=self.params['nTest'], output_dim=32)(test_input)
		x = np.concatenate((x, test_auxin, attr_input), axis=1)
		
		x = Dense(512, activation="relu")(x)
		x = Dense(512, activation="relu")(x)
		x = Dense(512, activation="relu")(x)		
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

#TODO: LR, 
		

