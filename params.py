##################################################################
#
# Data parameters
#
##################################################################

batch_size = 128
data_folder = "./data"
num_classes = 100

##################################################################
#
# Learning process parameters
#
##################################################################

learning_rate = 0.01
learning_rate_decay_epoch = 10
learning_rate_decay_coefficient = 0.5
momentum = 0.9
number_of_epochs = 2000
skip_step = 2000
name_prefix_for_saved_model = 'model-'
mode_classification = 'classification'
mode_representation = 'representation'
