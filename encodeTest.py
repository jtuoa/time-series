import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import category_encoders as ce
import pdb

#load both datasets
dataset1 = pd.read_csv("WoundInf_Train_Tests.tsv", delimiter='\t', encoding='utf-8')
dataset1.fillna(0, inplace = True)
x1 = dataset1.loc[:,['PID', 'Date', 'TestType', 'NumAnswer']]
x1 = np.array(x1)
x1 = x1[0:10,:] #for test

words = x1[:,2]

#onehot encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(words)
onehot_encoded = to_categorical(integer_encoded)

#binary encode
#requires import categorical_encoder as ce
#this isn't installed on server

#hashing encode

pdb.set_trace()
print(x1[0:3,:])
