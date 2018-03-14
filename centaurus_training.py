import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import binary_classification_learning
import omniglot
import params
import similarity_network_effective
import utils


def centaurus():
    ##################################################################
    #
    # Loading data images
    #
    ##################################################################
    train_loader, test_loader = omniglot.download_Omniglot_for_representation(data_folder='')

    ##################################################################
    #
    # Create a network for classification pretrainig and representations learning
    #
    ##################################################################
    representation_length = 105 * 105
    print('representation_length = ', representation_length)
    network = similarity_network_effective.EffectiveSimilarityNetwork(
        number_of_input_features=representation_length, l1_initialization=False).cuda()

    ##################################################################
    #
    # Do pairs learning
    #
    ##################################################################

    restore_epoch = params.default_recovery_epoch_for_binary_classification
    optimizer = optim.SGD(network.parameters(),
                          lr=params.learning_rate_for_binary_classification,
                          momentum=params.momentum_for_binary_classification)
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    if params.recover_binary_classification:
        print('Restore for classification pre-training')
        restore_epoch = params.default_recovery_epoch_for_classification
        network, optimizer = utils.load_network_and_optimizer_from_checkpoint(network=network,
                                                                              optimizer=optimizer,
                                                                              epoch=restore_epoch,
                                                                              name_prefix_for_saved_model=
                                                                              params.name_prefix_for_saved_model_for_binary_classification)
        start_epoch = restore_epoch
    else:
        start_epoch = 0

    print('Create a multi_lr_scheduler')
    # Decay LR by a factor of 0.1 every 10 epochs
    multi_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[82, 123],
                                                  gamma=0.1)

    ##################################################################
    #
    # Classification
    #
    ##################################################################
    print('Start binary classification ')
    binary_classification_learning.binary_learning(train_loader=train_loader,
                                  network=network,
                                  criterion=nn.CrossEntropyLoss(),
                                  test_loader=test_loader,
                                  optimizer=optimizer,
                                  start_epoch=start_epoch,
                                  lr_scheduler=multi_lr_scheduler)


centaurus()