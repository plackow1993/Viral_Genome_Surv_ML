######## Neural Network for base calling - training models step to create most accurate model #########

import numpy as np
import pandas as pd
import random
import sys
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.pipeline import Pipeline
from keras.models import load_model
import time
import middle_select as MS
import consensus as CS
import math

start_time = time.time()

random.seed(0)



#load training and testing data (added ../ to keep the data files in the orginal directory, remove if they are in the same directory)

#print(len(training_data))
test_data = pd.read_csv('test_data_new2.csv')
test_data = test_data.iloc[:,1:]

#establish the base list, must be in alphabetical order, which you can also input to the encoder for each option (if we want to consider deletions that will come first alphabetically, just a note for future reference) this WILL give us N's if necessary but some care will need to be taken:
base_list = ['A', 'C', 'G', 'N', 'T']
encoder = LabelEncoder()
encoder.fit(base_list)
#encoder is now ready to be used and training data can be fully manipulated.




## Adding in targets for training and testing.
#Parameters for target_id --> 1 is an adjusted target: 2 is the reference base target: 3-8 are base calls sense and antisense at increasing exposures.

test_target = test_data.iloc[:, 1]
test_target_later = test_data.iloc[:,1]

## encode class values as integers

encoded_targets_test = encoder.transform(test_target)



# convert integers to dummy variables (i.e. one hot encoded)
test_target = np_utils.to_categorical(encoded_targets_test)

#the target variables in column order are in alphabetical order, ACGNT

#Need to normalize and transform to numpy array the training and testing data

test_data = test_data.iloc[:,9:]
#print(training_data)
#print(test_data)




test_data = normalize(test_data, norm = 'l2', axis=0)



### Build the neural network model here:
#Loas in Model
model = load_model('saved_models/saved_model_00000029846_0000000relu_00000000500_00000005000_00000000005_consensusH6_00000000012')
model.summary()
quit()

# Make predictions on test set. prediction probs accurately represents probabilitiesof each base call and argmax(predictions_probs) gives the numerical index of the maximum likelihood call. Predicted bases gives the base call letter (ATCGN) at each position in the test dataset.
predictions_probs = model.predict(test_data)
print(predictions_probs)
predictions_list = []
for i in range(0, len(predictions_probs)):
    predictions_list.append(np.argmax(predictions_probs[i]))

predicted_bases = encoder.inverse_transform(predictions_list)

count_correct = 0
for i in range(0, len(predicted_bases)):
    if test_target_later[i] == predicted_bases[i]:
        count_correct += 1
acc = count_correct/test_data.shape[0]
#print(acc)

#lines for writing to a text files with the accuracies


acc = round(acc,9)

#final line for parallel printing
print("accuracy is = ", acc)
#print('middle_kept =', middle_kept, ", activation function =",act, ', epochs =', epochs, ', batch size =', batch_size, ', neurons = ', neurons, ', gives an accuracy of', acc)





