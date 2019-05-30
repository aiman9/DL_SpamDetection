import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import inputfiles

#load data and files
X,Y=inputfiles.load_data_and_labels_from_files('../Spam_Detection_Data/')
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
max_words = 10000
# max_len = 800

#tokenize documents and convert to sequences
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)

#padding sequences
max_len=max(len(sequences[i]) for i in range(len(sequences)))
sequences_matrix = sequence.pad_sequences(sequences, padding='post',truncating='post', maxlen=max_len)

print("Max_words={}\nMax_len={}".format(max_words,max_len))

def RNN():
    inputs = Input(name='inputs',shape=[sequences_matrix.shape[1]])
    layer = Embedding(max_words,300,input_length=sequences_matrix.shape[1],mask_zero=True)(inputs)
    layer = LSTM(150)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])
model.fit(sequences_matrix,Y_train,batch_size=32,epochs=10,
          validation_split=0.1)

#prepare test data for evaluation
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,padding='post',truncating='post',maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))