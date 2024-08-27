#Combine neural net loading and stacking meta models to evaluate final set of wyoming validation data.

import numpy as np
import pandas as pd
import random
import sys
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.utils import np_utils
from sklearn.pipeline import Pipeline
from keras.models import load_model
import time
import math
from sklearn.preprocessing import LabelEncoder, normalize, OneHotEncoder
from sklearn.model_selection import StratifiedKFold as Skfold
from sklearn.linear_model import LogisticRegression
import os
import pickle

random.seed(0)

#handle the log_reg loading and model loading typing
log_reg_model = sys.argv[2]
log_reg_model = log_reg_model[9:]
loaded_reg_model = pickle.load(open(sys.argv[2], 'rb'))

if 'final' in log_reg_model:
        cutoff = 0
        log_reg_stat = 'final'
elif '99' in log_reg_model:
        cutoff = 233
        log_reg_stat = log_reg_model[11:]
elif 'top_100' in log_reg_model:
        cutoff = 411
        log_reg_stat = '100'
elif 'top_10_' in log_reg_model:
        cutoff = 501
        log_reg_stat = '10'
elif 'top_25' in log_reg_model:
        cutoff = 486
        log_reg_stat = '25'
elif 'top_50' in log_reg_model:
        cutoff = 461
        log_reg_stat = '50'
 
#load in models to be used in stacking
model_names = []
model_file = open('model_names.txt', 'r')

model_numbers = []

for count, file in enumerate(model_file):
	#print(file[:-1], count+1) #count <=234
	if count >= cutoff and count != 511:
		model_names.append(file[:-1])
		model_numbers.append(count+1)

#load validation datasets (do one by one for evaluation)
set_name = sys.argv[1]
valid_data = pd.read_csv('validation_sets/{}'.format(set_name))
valid_data = valid_data.fillna(0)
#print(valid_data)
position = valid_data.iloc[:,1]
valid_data = valid_data.iloc[:,2:]

print(len(model_numbers), len(model_names))

#establish the base list, must be in alphabetical order, which you can also input to the encoder for each option (if we want to consider deletions that will come first alphabetically, just a note for future reference) this WILL give us N's if necessary but some care will need to be taken:
base_list = ['A', 'C', 'G', 'N', 'T']
encoder = LabelEncoder()
encoder.fit(base_list)
#encoder is now ready to be used and training data can be fully manipulated.

## Adding in targets for training and validing.
#Parameters for target_id --> 1 is an adjusted target: 2 is the reference base target: 3-8 are base calls sense and antisense at increasing exposures.

valid_target = valid_data.iloc[:, 0]
valid_target_later = valid_data.iloc[:,0]

## encode class values as integers

encoded_targets_valid = encoder.transform(valid_target)



# convert integers to dummy variables (i.e. one hot encoded)
valid_target = np_utils.to_categorical(encoded_targets_valid)

#the target variables in column order are in alphabetical order, ACGNT

#Need to normalize and transform to numpy array the training and validing data
#for validation, you do not have the test_targets option. So this will retain all the information you need.
valid_data = valid_data.iloc[:,7:]

valid_data = normalize(valid_data, norm = 'l2', axis=0)

os.chdir('../ML_saving')
######## Create the stacking meta-model here with the model dictionary above ###########

predicted_stack = None

for loaded_model in model_names:
    ### Make predictions on test set. prediction probs accurately represents probabilities of each base call and argmax(predictions_probs) gives the numerical index of the maximum likelihood call. Predicted bases gives the base call letter (ATCGN) at each position in the test dataset.
    
    model = load_model('saved_models/{}'.format(loaded_model))
    predictions_probs = model.predict(valid_data)
    
    predictions_list = []
    for i in range(0, len(predictions_probs)):
        predictions_list.append(np.argmax(predictions_probs[i]))
    
    predicted_bases = np_utils.to_categorical(predictions_list)
    
    # creating training features consisting of model output results per chosen model
    if predicted_stack is None:
        predicted_stack = predictions_probs
    else:
        predicted_stack = np.dstack([predicted_stack, predictions_probs])

if cutoff == 0:
    new_model_list = []
    for model in model_names:
        for base in base_list:
            weights = model +'_'+base+'_weight'
            new_model_list.append(weights)
    model_list = new_model_list

elif cutoff > 0:
    new_model_list = []
    for number in model_numbers:
        for base in base_list:
            weights = str(number) + '_' + base
            new_model_list.append(weights) 
    model_list = new_model_list

#this creates a dataframe of the stacked data to be sent through the selected model for logistic regression per sample.
dataframe_stacked = pd.DataFrame(columns = model_list, index = range(0,predicted_stack.shape[0]))
entry_counter = 0
for n in predicted_stack:
    m = np.transpose(n)
    model_counter = 0
    for M in m:
        dataframe_stacked.iloc[entry_counter, 5*model_counter:5*(model_counter+1)] = M
            #dataframe_stacked.iloc[entry_counter]
        model_counter += 1
    entry_counter += 1
print(dataframe_stacked)
pre_NN_name = '{}_stacked_weights_for_NN.csv'.format(set_name[0:2])
dataframe_stacked.to_csv(pre_NN_name)
quit()
#sending the stacked dataframe through the logistic regression model. Have to load it in first and possible redirect towards directory with the models inside.
log_reg_predicts = loaded_reg_model.predict_proba(dataframe_stacked)
columns_log_reg = ['A_prob', 'C_prob', 'G_prob', 'T_prob']
final_predictions = pd.DataFrame(log_reg_predicts, columns = columns_log_reg)
csv_name = log_reg_stat + set_name 
os.chdir('../ML_validation')
final_predictions.to_csv(csv_name)

