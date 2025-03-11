
# Logistic Regression as our meta model
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, normalize, OneHotEncoder
from sklearn.model_selection import StratifiedKFold as Skfold
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
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


#separate test data into a train test split to verify stacked model performance
stack_train_X, stack_test_X, stack_train_Y, stack_test_Y = tts(test_data, test_target, train_size = 0.5, random_state = 1)
print(stack_train_X.shape, stack_test_X.shape, stack_train_Y.shape, stack_test_Y.shape)


stack_train_Y = stack_train_Y.to_list()
stack_test_Y = stack_test_Y.to_list()




penalty = 0.12
meta_learner = LogisticRegression(max_iter = 100000, multi_class = 'multinomial', tol = 0.01, C = penalty )

meta_learner_model = meta_learner.fit(stack_train_X, stack_train_Y)


file_name = '../../saved_metas_stacking/log_reg_meta.pickle'
pickle.dump(meta_learner, open(file_name, "wb"))

#### This is code for running the validation studies. We did studies on which models to stack (thats the testing stack portion, remove that for loop for log reg parameter searching. But ultimately decided on the top models that had 99% or better. So this is most useful for testing the tolerance and penalty values. Will comment out for running the program with the correct dataset and training a final model
#for stack in stack_list:
#        testing_stack = stack
#        skf = Skfold(n_splits = 10, shuffle = True, random_state=1993)
#        for penalty in penalty_list:
#                for tol in tol_list:
#                        meta_learner = LogisticRegression(max_iter = 100000, multi_class = 'multinomial', tol = tol, C = penalty )
#
#                        strat_scores = []
#
#                        for training_index, test_index in skf.split(testing_stack.iloc[:,0:-1],testing_stack.iloc[:,-1]):
#                                train_X = testing_stack.iloc[training_index, 0:-1]
#                                train_Y = testing_stack.iloc[training_index, -1]
#                                test_X = testing_stack.iloc[test_index, 0:-1]
#                                test_Y = testing_stack.iloc[test_index, -1]
#
#                                meta_learner_model = meta_learner.fit(train_X, train_Y)
#                                test_predictions_meta_learner = meta_learner_model.predict(test_X)
#                                count = 0
#                                for index, test in enumerate(test_predictions_meta_learner):
#                                        if test_Y.iloc[index] == test:
#                                                count += 1
#                                strat_scores.append(count/test_X.shape[0])
#
#
#                        print('\nMaximum Accuracy from {}, {}, and {}:'.format(str(stack),penalty,tol ), max(strat_scores)*100, '%')
#                        print('\nMinimum Accuracy:',  min(strat_scores)*100, '%')
#                        print('\nOverall Accuracy:', np.mean(strat_scores)*100, '%')
#                        print('\nStandard Deviation is:', np.std(strat_scores))

# use for saving multiple models with multiple parameters. Unnecssary for publication.
#stack_name = stack_name[:-4]
#file_name = '{}_{}_{}.pickle'.format(stack_name, penalty, tol)
#pickle.dump(meta_learner, open(file_name, "wb"))
