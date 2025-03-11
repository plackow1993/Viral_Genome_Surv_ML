# Use to rewrite the final stacking input data such that we save on names and fit towards the feature names from the saved meta networks

import pandas as pd
import numpy as np


def update_training_set_renaming():
    data = pd.read_csv('../../datasets/final_models_stacked_ordered.csv', index_col = 0)
    model_num = 1
    new_col = []
    my_list = []
    for col in data.columns[:-1]:
        base = col.split('_weight')[0].split('_')[-1]
        new_col.append('{}_{}'.format(model_num, base))
        if base == 'T':
            model_num += 1
            my_list.append('{}->{}'.format(model_num-1, col.split('_T_weight')[0]))

    new_col.append('targets')
    data.columns = new_col
    data.to_csv('../../datasets/final_models_stacked_ordered_renamed.csv')

    file_path = "../../datasets/number_to_name_mapping.txt"

    with open(file_path, "w") as file:
        for item in my_list:
            file.write(item + "\n")

    return

def make_training_set():
    data = pd.read_csv('../../datasets/final_models_stacked_ordered_renamed.csv', index_col = 0)
    data = data.iloc[:,5*233:]
    data.to_csv('../../datasets/meta_training_data.csv')
    
    return

make_training_set()
