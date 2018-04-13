import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import omniglot
import utils
from datasets.loaders import omniglot
from networks_and_layers.convolution_spoc_metric_network import ConvolutionSpocMetric
from training_procedures import binary_classification_learning
from utils import params


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
    network =  ConvolutionSpocMetric(
        desired_dimension_for_spoc=111,
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
