import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models

import omniglot
import utils
from datasets.loaders import omniglot
from networks_and_layers.convolution_spoc_metric_network import ConvolutionSpocMetric
from training_procedures import binary_classification_learning
from utils import metric_learning_utils, params
from utils.spoc import compute_spoc_by_outputs, learn_PCA_matrix_for_spocs_with_sklearn
from utils.utils import get_layer_with_number


def centaurus():
    ##################################################################
    #
    # Loading data images
    #
    ##################################################################
    train_loader, test_loader = omniglot.download_Omniglot_for_representation(
        data_folder='/media/natasha/Data/course-work-data/',
        image_size=128
    )


    ##################################################################
    #
    # Create a network for binary classification learning
    #
    ##################################################################
    desired_dimension = 128
    layer_number_in_convolutional_part = 11

    #############################################################################
    # Compute PCA matrix usijng all the data for initialization of the PCA matrix
    #############################################################################
    alexnet = get_layer_with_number(
            network=models.alexnet(pretrained=True),
            layer_number=layer_number_in_convolutional_part
        )

    if False:
        all_outputs, all_labels = metric_learning_utils.get_all_outputs_and_labels_for_large_dataset(train_loader, alexnet)
        print('all_outputs ', all_outputs.shape)
        all_spocs = compute_spoc_by_outputs(Variable(all_outputs))
        print('all_spocs ', all_spocs.data.shape)
        #PCA_matrix, singular_values = learn_PCA_matrix_for_spocs(all_spocs[:5000], desired_dimension)
        #print('initail PCA_matrix without sklearn', PCA_matrix.shape)
        #print('initail singular_values without sklearn', singular_values.shape)

        PCA_matrix, singular_values = learn_PCA_matrix_for_spocs_with_sklearn(all_spocs.data, desired_dimension)
        np.save('initial_PCA_matrix',PCA_matrix)
        np.save('initial_singular_values', singular_values)
        all_outputs = None
        all_labels = None
        all_spocs = None
        gc.collect()
    else:
        PCA_matrix = np.load('initial_PCA_matrix.npy')
        singular_values = np.load('initial_singular_values.npy')

    print('initial PCA_matrix', PCA_matrix.shape)
    print('initial singular_values', singular_values.shape)
    #############################################################################
    # Create a network with the proper initialization of PCA matrix
    #############################################################################
    network =  ConvolutionSpocMetric(
        desired_dimension_for_spoc=desired_dimension,
        initial_PCA_matrix=torch.from_numpy(PCA_matrix),
        initial_singular_values=torch.from_numpy(singular_values),
        layer_number_in_convolutional_part=layer_number_in_convolutional_part,
        network_for_convolutional_part=models.alexnet(pretrained=True),
        number_of_output_neurons=2
    ).cuda()

    ##################################################################
    #
    # Do pairs learning
    #
    ##################################################################

    optimizer = optim.Adam(
        network.parameters(),
        lr=params.learning_rate_for_binary_classification
    )
    print('Create a multi_lr_scheduler')
    # Decay LR by a factor of 0.1
    multi_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[82, 123],
        gamma=0.1
    )
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    if params.recover_binary_classification:
        print('Restore for binary classification')
        restore_epoch = params.default_recovery_epoch_for_binary_classification
        network, optimizer = utils.load_network_and_optimizer_from_checkpoint(
            network=network,
            optimizer=optimizer,
            epoch=restore_epoch,
            name_prefix_for_saved_model=params.name_prefix_for_saved_model_for_binary_classification
        )
        start_epoch = restore_epoch + 1
    else:
        start_epoch = 0

    ##################################################################
    #
    # Binary classification
    #
    ##################################################################
    print('\n\nStart binary classification ')
    binary_classification_learning.binary_learning(
        train_loader=train_loader,
        network=network,
        criterion=nn.CrossEntropyLoss(),
        test_loader=test_loader,
        optimizer=optimizer,
        start_epoch=start_epoch,
        lr_scheduler=multi_lr_scheduler
    )


centaurus()
