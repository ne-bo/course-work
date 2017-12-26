##################################################################
#
# Data parameters
#
##################################################################

batch_size = 16
number_of_samples_with_the_same_label_in_the_batch = 4
data_folder = "./data"
num_classes = 200

dataset = 'birds' # possible values 'cifar', 'birds'
network = 'resnet-50' # possible values 'small-resnet', 'resnet-50'

##################################################################
#
# Learning process parameters
#
##################################################################

learning_rate = 0.1
learning_rate_decay_epoch = 10
learning_rate_decay_coefficient = 0.5
momentum = 0.9
number_of_epochs = 300
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
learn_classification = False
default_recovery_epoch_for_classification = 160

recover_classification_net_before_representation = True
recover_representation_learning = False
default_recovery_epoch_for_representation = 160
