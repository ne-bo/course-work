##################################################################
#
# Data parameters
#
##################################################################

batch_size_for_cifar = 128

batch_size_for_classification = 28 # 28 is a maximum batch size for 224 x 224 images and 6 Gb of GPU memory
batch_size_for_representation = 28
# for equal number of positive and negative examples
number_of_samples_with_the_same_label_in_the_batch = (batch_size_for_representation + 1)/2
batch_size_for_similarity = 128

data_folder = "./data"
num_classes = 200

dataset = 'birds' # possible values 'cifar', 'birds'
network = 'resnet-50' # possible values 'small-resnet', 'resnet-50'

##################################################################
#
# Learning process parameters
#
##################################################################

learning_rate_for_classification = 0.01
learning_rate_for_representation = 0.01
learning_rate_for_similarity = 0.0001

learning_rate_decay_epoch = 10
learning_rate_decay_epoch_for_representation = 10

learning_rate_decay_coefficient = 0.5
learning_rate_decay_coefficient_for_representation = 0.5
learning_rate_decay_coefficient_for_similarity = 0.0005

momentum_for_classification = 0.9
momentum_for_similarity = 0.9


number_of_epochs_for_classification = 101
number_of_epochs_for_representation = 101
number_of_epochs_for_metric_learning = 101

skip_step = 100

name_prefix_for_saved_model_for_classification = 'model-cl-'
name_prefix_for_saved_model_for_representation = 'model-rp-'
name_prefix_for_similarity_saved_model = 'similarity-model-'

mode_classification = 'classification'
mode_representation = 'representation'

k_for_recall = 8


delta_for_similarity = 0.5

##################################################################
#
# Main flow parameters
#
##################################################################

recover_classification = False
learn_classification = False
default_recovery_epoch_for_classification = 100

recover_classification_net_before_representation = True
recover_representation_learning = True
default_recovery_epoch_for_representation = 90
learn_representation = True
