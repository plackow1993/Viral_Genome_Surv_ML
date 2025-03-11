# Public Repository for "Viral Genome Surveillance via Modifiable Microarray Sequencing and a Supervised Stack Ensemble Neural Network Model: SARS-CoV-2 as a Case Study"

The models used individually (top 278 as indicated in the paper) are saved in "saved_models".

The meta-models from stacking (logistic regression and neural network) are saved in "saved_metas_stacking".

Use "ML_saving.py" to train the models that get saved into "saved_models". Although "saved_models" aready contains the models used in stacking. So edits will needed to be made if you add any saved models. Run this using a (linux) terminal input as:  python ML_saving.py cut_ends activation neurons epochs batch_size depth consensus_type

    example -> python ML_saving.py 0 relu 10 500 200 5 consensusH4
    
    the example will train a model with training data from the consensusH4 scheme with 0 examples cut off from each end. Each hidden layer will use the relu activation function at a depth of 5 (input layer, 10 neurons, 20 neurons, 10 neurons, output layer), and will run for 500 epochs. The output will be a saved model with those parameters.
    
"NN_stacking.py" is used to train the neural network meta model for stacking. With the note above, you'll have to change the input layer shape to 5 X number of models.

"Log_reg_stacking.py" is used to train the logistic regression model for stacking. Same input scenario as in "NN_stacking.py" 

"stacking_data_prep.py" contains code to change the names of the final models dataframe for quicker loading. This will create headers that match the feature headers of the saved meta models (causes problems with some versions of keras). It also selects the top 278 as indicated in the paper -> creates a training set to use in the meta-training, called "meta_training_data.csv".
