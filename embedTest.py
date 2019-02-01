import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import pdb


# Define documents
docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!',
        'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']
#pdb.set_trace()
# Define class labels
#sentiment analysis problem = expressing ones opinion
#all first 5 words are positive
#last 5 words are negative
#for our blood test we need to create label
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

#integer encode the documents
pdb.set_trace()
own_embedding_vocab_size = 10 #can select >10 (min words/phrases)
encoded_docs_oe = [one_hot(d, own_embedding_vocab_size) for d in docs]
print(encoded_docs_oe)

#pad each sequence to ensure they are of the same length
max_length = 4 #since longest sequence has 4 words
padded_docs_oe = pad_sequences(encoded_docs_oe, maxlen=max_length, padding='post')
print("padded docs: ")
print(padded_docs_oe)

#define model
model = Sequential()
model.add(Embedding(input_dim = own_embedding_vocab_size, output_dim=32, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
#train model
model.fit(padded_docs, labels, epochs=50, verbose=0)
#evaluate model
loss, accuracy = model.evaluate(padded_docs_oe, labels, verbose=0)
print('Accuracy: %f' %(accuracy*100)) #obviously 100% since test data = train data
