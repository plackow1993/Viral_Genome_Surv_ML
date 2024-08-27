'''
Fuction to create a set of targets using a level of consensus. Where consensus is acheived if a certain number of calls (determined by the user) out of 6 agree. This consensus file will only invoke target creation. This will remove non-consensus targets from our training data file altogether.

Hybrid consensus will be employed by adding an H before the consensus number. This will keep the reference bases and simply CHANGE the consensus values. (Note, this may or may not change the reference base, since consensus between all 6 base calls could still agree with the reference base.
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize
from keras.utils import np_utils
import numpy
import math
import middle_select as MS

#####---- All of this gets taken care of in the ML_neural program.
#str = 'consensus6'
##establish the base list must be in alphabetical order which you can also input to the encoder for each option (if we want to consider deletions that will come first alphabetically, just a note for future reference):
base_list = ['A', 'C', 'G', 'N', 'T']
encoder = LabelEncoder()
encoder.fit(base_list)
#
#
#
##picks the consensus level from the user input.
#C=int(str[-1])
#
##loads training_data as input
#training_data = pd.read_csv('../training_data.csv')
#
#
#
#sublist = MS.subset_list(14800, 29846, 26)
#training_data = training_data.iloc[sublist,:]
#
##condenses the targets from the training_data file
#target_options = training_data.iloc[:,1:8]
#
#print(target_options.shape[1])
############ ----- function is found below. the above is for testing purposes only.

'''
Function begins here: Using this to establish consensus
consensus_type = level of consensus. Currently this is 4, 5, or 6. The entry format is consensusC.
target_options is the training_data dataframe's first 6 target options to output a new training_data file with consensus.
training_data is the starting dataframe to be updated with this function.

the other two variables will never change, but you must always type them in.
'''


def consensus(target_options, training_data, consensus_type, encoder, base_list):

    C = int(consensus_type[-1])
    hybrid_check = consensus_type[-2]
    
    if C < 4 and  C > 0:
        print('consensus levels below 4 are not yet developed, choose 4, 5 or 6')
        print('however, if you want to use simply the reference bases as targets use consensus0')
        quit()
    
    elif C==0:
        training_data.insert(1, 'targets', training_data.iloc[:,1])

    else:
        #create empty consensus list and target list for updating the training data.
        consensus_list = []
        target_list = []
        for i in range(0, target_options.shape[0]):
            #have to transform EACH set of values for each position
            encoded_values = encoder.transform(target_options.iloc[i,1:])
            encoded_value_matrix = np_utils.to_categorical(encoded_values)

            #normalization will establish consensus. Each sum will be equal to n*1/sqrt(n) where n is the value of consensus you prefer.
            normal_encoded = normalize(encoded_value_matrix, axis=0)
           
            MAX_SUM = max(numpy.sum(normal_encoded, axis=0))
            
            if MAX_SUM > (C*(1/math.sqrt(C))-0.01):
                min_non_zero = numpy.min(normal_encoded[numpy.nonzero(normal_encoded)])
                min_loc = numpy.where(normal_encoded == min_non_zero)[1][0]
                consensus_list.append(i)
                target_list.append(base_list[min_loc])
                
        hybrid_check = consensus_type[-2]
        if hybrid_check == 'H':
            training_data.insert(1, 'targets', training_data.iloc[:,1])
            training_data.iloc[consensus_list, 1] = target_list
            
        else:
            training_data = training_data.iloc[consensus_list, :]
            training_data.insert(1, 'targets', target_list)
    
    return training_data



##Transforming the test set into a more representative accurate test set. IE if 6 level consensus agrees with reference base, then keep in test set. only run once.
#
#test_data = pd.read_csv('../test_data.csv')
#print(test_data.shape)
#target_options = test_data.iloc[:,1:8]
#consensus_type = 'j6'
#test_data_new = consensus(target_options, test_data, consensus_type, encoder, base_list)
#print(test_data_new.shape[0])
#save_list = []
#for i in range(0, test_data_new.shape[0]):
#    if test_data_new.iloc[i,1] == test_data_new.iloc[i,2]:
#        save_list.append(i)
#test_data_new = test_data_new.iloc[save_list, :]
#test_data_new.to_csv('test_data_new.csv')
