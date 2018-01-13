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

learning_rate = 0.01
learning_rate_decay_epoch = 10
learning_rate_decay_coefficient = 0.5
momentum = 0.9

number_of_epochs_for_classification = 101
number_of_epochs_for_representation = 101

skip_step = 130

name_prefix_for_saved_model_for_classification = 'model-cl-'
name_prefix_for_saved_model_for_representation = 'model-rp-'

mode_classification = 'classification'
mode_representation = 'representation'

k_for_recall = 8


number_of_epochs_for_metric_learning = 100
delta_for_similarity = 0.5
name_prefix_for_similarity_saved_model = 'similarity-model-'

##################################################################
#
# Main flow parameters
#
##################################################################

recover_classification = False
learn_classification = False
default_recovery_epoch_for_classification = 100

recover_classification_net_before_representation = False
recover_representation_learning = False
default_recovery_epoch_for_representation = 100
learn_representation = False
