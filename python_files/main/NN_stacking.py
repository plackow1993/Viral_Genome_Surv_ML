## Neural network as our meta model
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, normalize, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
#from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import math
import os
#import matplotlib as plt
np.set_printoptions(linewidth=200, edgeitems = 5)


########## Load and preprocess testing data to put into stacked model.
test_data = pd.read_csv("../../datasets/meta_training_data.csv")
test_target = test_data.iloc[:,-1]

test_data = test_data.iloc[:,1:-1]


#establish the base list, must be in alphabetical order, which you can also input to the encoder for each option (if we want to consider deletions that will come first alphabetically, just a note for future reference) this WILL give us N's if necessary but some care will need to be taken:
base_list = ['A', 'C', 'G', 'N', 'T']
encoder = LabelEncoder()
encoder.fit(base_list)
#encoder is now ready to be used and training data can be fully manipulated.

reminder_results = encoder.transform(base_list)

##creating a one hot encoding to match with neural network results
#reminder_results = np_utils.to_categorical(reminder_results)


#separate test data into a train test split to verify stacked model performance
stack_train_X, stack_test_X, stack_train_Y, stack_test_Y = tts(test_data, test_target, train_size = 0.5, random_state = 1)
print(stack_train_X.shape, stack_test_X.shape, stack_train_Y.shape, stack_test_Y.shape)


stack_train_Y = stack_train_Y.to_list()
stack_train_Y = to_categorical(stack_train_Y)
stack_test_Y = stack_test_Y.to_list()
stack_test_Y = to_categorical(stack_test_Y)


neuron_numbers = [2, 2]
act = 'selu'
neurons = [30]


for neuron in neurons:
    model = Sequential()
    model.add(Dense(neuron, input_dim=1390, activation=act))
    model.add(Dense(neuron, activation = act))
    model.add(Dense(neuron, activation = act))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.fit(stack_train_X, stack_train_Y, epochs=250, batch_size=200, validation_data = (stack_test_X, stack_test_Y), verbose = 1)
    
    
    model.save('../../saved_metas_stacking/NN_model_stacked_accuracy_rewrite')
