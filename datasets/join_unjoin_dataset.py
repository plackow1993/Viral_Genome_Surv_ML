# This is to rejoin training datasets, as well as split them. They can be quite large for download.
# delete the main csv file if you are separating to move around.

import pandas as pd
import numpy as np

# change the training step you want.  1- base training, 2- stacked training
train = 3

# change the version you want. 1- join, 2- separate
join = 2


if train == 1:
    string = 'wyup_train'
    num = 10
elif train == 2:
    string = 'final_models_stacked_ordered'
    num = 60
elif train == 3:
    string = 'meta_training_data'
    num = 40
    

if join == 1:
    fill_frame = pd.DataFrame()
    for val in range(0,num):
        fill_frame = pd.concat([fill_frame, pd.read_csv('{}_{}.csv'.format(string, val), index_col = 0)], axis = 0)
    fill_frame.to_csv('{}.csv'.format(string))
    
elif join == 2:
    data = pd.read_csv("{}.csv".format(string), index_col = 0)
    data_sep = np.array_split(data, num)
    for count, data in enumerate(data_sep):
        data.to_csv('{}_{}.csv'.format(string, count))
        
    
    try:
        # will run if you renamed your files
        string = 'final_models_stacked_ordered_renamed'
        data2 = pd.read_csv("{}.csv".format(string), index_col = 0)
        data_sep = np.array_split(data2, num)
        for count, data in enumerate(data_sep):
            data.to_csv('{}_{}.csv'.format(string, count))
    except:
        pass
    
