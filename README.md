# Public Repository for "Viral Genome Surveillance via Modifiable Microarray Sequencing and a Supervised Stack Ensemble Neural Network Model: SARS-CoV-2 as a Case Study"
The models used individually (top 278 as indicated in the paper) are saved in "saved_models".
The meta-models from stacking (logistic regression and neural network) are saved in "saved_metas_stacking".
Use "ML_saving.py" to train the models that get saved into "saved_models". Although "saved_models" aready contains the models used in stacking. So edits will needed to be made if you add any saved models.
"NN_stacking.py" is used to train the neural network meta model for stacking. With the note above, you'll have to change the input layer shape to 5 X number of models.

