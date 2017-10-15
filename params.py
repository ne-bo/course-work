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
number_of_epochs = 165
skip_step = 130
name_prefix_for_saved_model = 'model-'
mode_classification = 'classification'
mode_representation = 'representation'

k_for_recall = 10

##################################################################
#
# Main flow parameters
#
##################################################################

recover_classification = False
learn_classification = True
defalut_recovery_epoch_for_classification = 160

recover_classification_net_before_representation = True
defalut_recovery_epoch_for_representation = 160
