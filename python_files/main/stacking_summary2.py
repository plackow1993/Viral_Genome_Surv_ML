######## Stacking model with weights. This will build a matrix of prob vectors for each model to then use for stacking. #########

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, normalize, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
import math
import os
import matplotlib as plt
np.set_printoptions(linewidth=200, edgeitems = 5)

penalty = 0.01


########## Load and preprocess testing data to put into stacked model.
test_data = pd.read_csv('wyup_fixed_test.csv')
test_data = test_data.iloc[:,1:]


#establish the base list, must be in alphabetical order, which you can also input to the encoder for each option (if we want to consider deletions that will come first alphabetically, just a note for future reference) this WILL give us N's if necessary but some care will need to be taken:
base_list = ['A', 'C', 'G', 'N', 'T']
encoder = LabelEncoder()
encoder.fit(base_list)
#encoder is now ready to be used and training data can be fully manipulated.

## Adding in targets testing
test_target = test_data.iloc[:, 1]
test_target_later = test_data.iloc[:,1]

# encode class values as integers
encoded_targets_test = encoder.transform(test_target)

reminder_results = encoder.transform(base_list)

##creating a one hot encoding to match with neural network results
#reminder_results = np_utils.to_categorical(reminder_results)


#actual test data
test_data = test_data.iloc[:,9:]

#separate test data into a train test split to verify stacked model performance
stack_train_X, stack_test_X, stack_train_Y, stack_test_Y = tts(test_data, test_target, train_size = 0.999999, random_state = 1)
#print(stack_train_X.shape, stack_test_X.shape, stack_train_Y.shape, stack_test_Y.shape)


stack_train_Y = stack_train_Y.to_list()
stack_test_Y = stack_test_Y.to_list()

stack_train_Y = encoder.transform(stack_train_Y)
stack_test_Y = encoder.transform(stack_test_Y)
stack_train_Y_categorical = np_utils.to_categorical(stack_train_Y)

#print(stack_train_Y_categorical)
# convert integers to dummy variables (i.e. one hot encoded)
#test_target = np_utils.to_categorical(encoded_targets_test)
stack_train_X = normalize(stack_train_X, norm = 'l2', axis=0)
stack_test_X = normalize(stack_test_X, norm = 'l2', axis=0)

######## Creates a dictionary that holds all model class instances to stack final results into our training set for our meta model. ###########

#creates a list of model directories to load and evaluate trained models.
os.chdir('saved_models2')
model_list = [name for name in os.listdir() if os.path.isdir(name)]


#Load in Model(s)
model_options = {}
for i, saved_model in enumerate(model_list):
    model = load_model(saved_model)
    model_options[saved_model] = [model]
    model_options[saved_model].append(saved_model)
    model_options[saved_model].append(i)



######## Create the stacking meta-model here with the model dictionary above ###########

predicted_stack = None

for model_name in model_list:
    ### Make predictions on test set. prediction probs accurately represents probabilities of each base call and argmax(predictions_probs) gives the numerical index of the maximum likelihood call. Predicted bases gives the base call letter (ATCGN) at each position in the test dataset.
    
    model = model_options[model_name][0]
    predictions_probs = model.predict(stack_train_X, verbose=0)
    
    predictions_list = []
    for i in range(0, len(predictions_probs)):
        predictions_list.append(np.argmax(predictions_probs[i]))
    
    predicted_bases = np_utils.to_categorical(predictions_list)
    
    #predicted_bases_probs = np.dstack([predicted_bases, predictions_probs])
    # creating training features consisting of model output results per chosen model
    if predicted_stack is None:
        predicted_stack = predictions_probs
    else:
        predicted_stack = np.dstack([predicted_stack, predictions_probs])
    
    #calculate accuracy of each individual model
    count_correct = 0
    for i in range(0, len(predicted_bases)):
        if stack_train_Y[i] == np.argmax(predicted_bases[i]):
            count_correct += 1
    acc = count_correct/stack_train_X.shape[0]
    acc = round(acc,9)
    
    #round out dictionary with indiviual model accuracies
    model_options[model_name].append(acc)

    print("The accuracy for training model {} is = ".format(model_name), acc)
    

print('done loading')





#creates an ordered dictionary of most accurate models. The index will help to order the matrix so that from left to right, you have the lowest to the highest accuracy on the test set.
model_options = sorted(model_options.values(), key=lambda value: value[3])

#creating new dictionary of the ordered models (by test data accuracy)
model_options_2 = {}
for list in model_options:
    model_options_2[list[1]] = [list[2],list[3]]

#checking the order for debugging
for value in model_options_2.values():
    print(value[0])

#create updated list for weight labels
model_list =[]
for key in model_options_2.keys():
    model_list.append(key)

#create list for creation of empy dataframe)
new_model_list = []
for model in model_list:
    for base in base_list:
        weights = model+"_"+base+'_weight'
        new_model_list.append(weights)
new_model_list.append('training_targets')

dataframe_stacked = pd.DataFrame(columns = new_model_list, index = range(0,predicted_stack.shape[0]))

print(dataframe_stacked)


#fill dataframe using the values where the predicted stack contains the data for each model.
for count, value in enumerate(model_options_2.values()):
    dataframe_stacked.iloc[:,5*count:5*(count+1)] = predicted_stack[:,:,value[0]]



dataframe_stacked['training_targets']=stack_train_Y


#just want to check the fill to see if its correct, comment out once confirmed
for count, model in enumerate(model_list):
    count_correct = 0
    for i in range(0, dataframe_stacked.shape[0]):
        if dataframe_stacked.iloc[i,-1] == np.argmax(dataframe_stacked.iloc[i,5*count:5*(count+1)]):
            count_correct += 1
    acc = count_correct/dataframe_stacked.shape[0]
    acc = round(acc,9)
    print(acc)


dataframe_stacked.to_csv("../final_models_stacked_ordered2.csv")



