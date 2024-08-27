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
import time
import middle_select as MS
import consensus as CS
import math

start_time = time.time()

random.seed(0)


#catching any erroneous inputs before we get started.
if len(sys.argv) != 9:
    print("you are missing one of your inputs when calling this function: python ML_neural.py cut_ends activation neurons epochs batch_size depth target_id consensus_type")
    quit()
    
print(sys.argv)
cut_ends = int(sys.argv[1])
act = sys.argv[2]
neurons = int(sys.argv[3])
epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])
depth = int(sys.argv[6])
target_id = int(sys.argv[7])
consensus_type = sys.argv[8]

if cut_ends > 14922 or cut_ends < 0:
    print("cut_ends value should be between 0 and 14922, inclusive, please run again with proper input")
    quit()

if depth < 2:
    print("base network depth is 2 (input and output). So your value for depth has to be 2 or greater")
    quit()


#load training and testing data (added ../ to keep the data files in the orginal directory, remove if they are in the same directory)
training_data = pd.read_csv('wyup_training_data.csv')
#print(len(training_data))
test_data = pd.read_csv('wyup_fixed_test.csv')
test_data = test_data.iloc[:,1:]

#print(test_data)
#establish the base list, must be in alphabetical order, which you can also input to the encoder for each option (if we want to consider deletions that will come first alphabetically, just a note for future reference) this WILL give us N's if necessary but some care will need to be taken:
base_list = ['A', 'C', 'G', 'N', 'T']
encoder = LabelEncoder()
encoder.fit(base_list)
#encoder is now ready to be used and training data can be fully manipulated.




"""
---------Preprocessing the training data-----------
Subset the training data in a list omitting certain ends of each dataset. This updates training data.

Function:   subset_list(n,N,M)
N = size of basepairs = 29846 = sample_#n
cut_ends or n = size of cutoff at ends. Can try anywhere from 0 (full set) to 14922 (middle 2 bases). This is defined within your python script from the terminal.
M = sets used for the training data. We started with 26 out of our original 33.


Then, create a consensus subset or a hybrid consensus edit to the reference base from which to train on.

Training_data is now a fully characterized dataset with an added target column located at the index 1 column of the dataframe. it is of shape M*(N-2*cut_ends) x 57
"""

sublist=MS.subset_list(cut_ends,29846,20)
training_data = training_data.iloc[sublist, :]

# target options is a matrix with possible targets. Used with consensus method in consensus.py file:
target_options = training_data.iloc[:,1:8]

training_data = CS.consensus(target_options, training_data, consensus_type, encoder, base_list)

#print("training_data shape is", training_data.shape)
del(target_options)


## Adding in targets for training and testing.
#Parameters for target_id --> 1 is an adjusted target: 2 is the reference base target: 3-8 are base calls sense and antisense at increasing exposures.
training_target = training_data.iloc[:,target_id]
test_target = test_data.iloc[:, 1]
test_target_later = test_data.iloc[:,1]

## encode class values as integers
encoded_targets_training = encoder.transform(training_target)
encoded_targets_test = encoder.transform(test_target)



# convert integers to dummy variables (i.e. one hot encoded)
training_target = np_utils.to_categorical(encoded_targets_training)
test_target = np_utils.to_categorical(encoded_targets_test)

#the target variables in column order are in alphabetical order, ACGNT

#Need to normalize and transform to numpy array the training and testing data
training_data = training_data.iloc[:,9:]
test_data = test_data.iloc[:,9:]
#print(training_data)
#print(test_data)



#
training_data = normalize(training_data, norm = 'l2', axis=0)
test_data = normalize(test_data, norm = 'l2', axis=0)


#Depending on the parity of layers, this outputs the powers of 2 that go into each neuron layer for an escalating neuron make-up. It will double the amount of start_neurons until it reaches the middle layer.
neuron_numbers = []
depth=depth-2
if depth == 1:
    neuron_numbers = [0]
    
elif (depth % 2) == 0:
    for i in range(0,math.floor(depth/2)):
        neuron_numbers.append(i)
    
    rev_numbers = list(reversed(neuron_numbers))
    for j in rev_numbers:
        neuron_numbers.append(j)
    
elif (depth % 2) != 0:
    for i in range(0,math.floor(depth/2)):
        neuron_numbers.append(i)
        rev_numbers = list(reversed(neuron_numbers))
    neuron_numbers.append(math.floor(depth/2))
    for j in rev_numbers:
        neuron_numbers.append(j)


### Build the neural network model here:

# define baseline model

# create model
model = Sequential()
for k in neuron_numbers:
    model.add(Dense(neurons*(2**k), input_dim=48, activation=act))
model.add(Dense(5, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#
#estimator = KerasClassifier(model=baseline_model, epochs=epochs, batch_size=batch_size, verbose=0)

model.fit(training_data, training_target, epochs=epochs, batch_size=batch_size)

### Make predictions on test set. prediction probs accurately represents probabilitiesof each base call and argmax(predictions_probs) gives the numerical index of the maximum likelihood call. Predicted bases gives the base call letter (ATCGN) at each position in the test dataset.
predictions_probs = model.predict(test_data)
#print(predictions_probs)
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
lines = []
lines.append('middle_kept = '+ str(2*(14923-cut_ends))+ 'activation = '+act+ ', epochs = '+str(epochs)+', batch = '+str(batch_size)+', neurons = '+str(neurons)+'gives an accuracy of ' +str(acc))

#lines to key in on parallel output
middle_kept = 2*(14923-cut_ends)
act = str(act).zfill(11)
middle_kept = str(middle_kept).zfill(11)
epochs = str(epochs).zfill(11)
batch_size = str(batch_size).zfill(11)
neurons = str(neurons).zfill(11)
consensus_type = str(consensus_type).zfill(11)
depth = depth+2
depth = str(depth).zfill(11)
acc = round(acc,9)

#final line for parallel printing
print(middle_kept, act, epochs, batch_size, neurons, consensus_type, depth, acc)
#print('middle_kept =', middle_kept, ", activation function =",act, ', epochs =', epochs, ', batch size =', batch_size, ', neurons = ', neurons, ', gives an accuracy of', acc)
model_name = 'saved_model_'+middle_kept + "_" + act  + "_" + epochs + "_" + batch_size + "_" + neurons + "_" + consensus_type + "_" + depth
if acc > 0.90:
    model.save('saved_models/'+model_name)
else:
    print(model_name + " does not have high enough accuracy to save model")
quit()



# can figure out printing later, so far the code works.
with open('consensus_accuracies.txt', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
        
end_time = time.time()
print("all data with batch size of 5", end_time-start_time, "seconds")
quit()


results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.10f%% (%.10f%%)" % (results.mean()*100, results.std()*100))

end_time = time.time()
print("all data with batch size of 5", end_time-start_time, "seconds")
