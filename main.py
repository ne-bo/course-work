from torch.autograd import Variable

import cifar
import learning
import metric_learning_utils
import similarity_network_effective
import spoc
import test
import params
import torchvision
import utils
import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import loss
from torch.optim import lr_scheduler
import small_resnet_for_cifar
import cProfile
import pstats
import io
import histogramm_loss
import birds
import torch.utils.model_zoo
import metric_learning
import similarity_network
import numpy as np


def debug_images_show(train_loader_for_classification):
    ##################################################################
    #
    # Images show for debug
    #
    ##################################################################
    # get some random training images
    dataiter = iter(train_loader_for_classification)
    images, labels = dataiter.next()
    # show images
    print("images.shape ", images.shape)
    utils.imshow(torchvision.utils.make_grid(images))  # images = Tensor of shape (B x C x H x W)
    # print labels
    print(' '.join('%5s' % labels[j] for j in range(params.batch_size)))


def classification_pretrainig(network, train_loader_for_classification, test_loader_for_classification):
    restore_epoch = params.default_recovery_epoch_for_classification
    optimizer = optim.SGD(network.parameters(),
                          lr=params.learning_rate_for_classification,
                          momentum=params.momentum_for_classification)
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    if params.recover_classification:
        print('Restore for classification pre-training')
        restore_epoch = params.default_recovery_epoch_for_classification
        network, optimizer = utils.load_network_and_optimizer_from_checkpoint(network=network,
                                                                              optimizer=optimizer,
                                                                              epoch=restore_epoch,
                                                                              name_prefix_for_saved_model=
                                                                              params.name_prefix_for_saved_model_for_classification)
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
    # Classification for pre-training
    #
    ##################################################################
    if params.learn_classification:
        print('Start classification pretraining')
        learning.learning_process(train_loader=train_loader_for_classification,
                                  network=network,
                                  criterion=nn.CrossEntropyLoss(),
                                  test_loader=test_loader_for_classification,
                                  all_outputs_test=None, all_labels_test=None,
                                  mode=params.mode_classification,
                                  optimizer=optimizer,
                                  start_epoch=start_epoch,
                                  lr_scheduler=multi_lr_scheduler)


def representations_learning(network, train_loader, test_loader):
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    if params.recover_classification_net_before_representation:
        print('Restoring before representational training')
        restore_epoch = params.default_recovery_epoch_for_classification
        network = utils.load_network_from_checkpoint(network=network,
                                                     epoch=restore_epoch,
                                                     name_prefix_for_saved_model=
                                                     params.name_prefix_for_saved_model_for_classification)

    ##################################################################
    #
    # Training for representations with our margin loss
    #
    ##################################################################

    print('Representational training')

    optimizer_for_representational_learning = optim.Adam(network.parameters(),
                                                         lr=params.learning_rate_for_representation)  # ,
    # momentum=params.momentum)

    if params.recover_representation_learning:
        print('Restore for representational learning')
        restore_epoch = params.default_recovery_epoch_for_representation
        network, optimizer_for_representational_learning = utils.load_network_and_optimizer_from_checkpoint(
            network=network,
            optimizer=optimizer_for_representational_learning,
            epoch=restore_epoch,
            name_prefix_for_saved_model=params.name_prefix_for_saved_model_for_representation)
        start_epoch = restore_epoch
    else:
        start_epoch = 0

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_for_representational_learning,
                                           step_size=params.learning_rate_decay_epoch_for_representation,
                                           gamma=params.learning_rate_decay_coefficient_for_representation)

    if params.learn_representation:
        learning.learning_process(train_loader=train_loader,
                                  network=network,
                                  # criterion=loss.MarginLoss(),
                                  criterion=histogramm_loss.HistogramLoss(150),
                                  # criterion=nn.CrossEntropyLoss(),
                                  test_loader=test_loader,
                                  all_outputs_test=None,
                                  all_labels_test=None,
                                  mode=params.mode_representation,
                                  optimizer=optimizer_for_representational_learning,
                                  lr_scheduler=exp_lr_scheduler,
                                  start_epoch=start_epoch)

        print("Evaluation: ")
        all_outputs_test, all_labels_test = metric_learning_utils.get_all_outputs_and_labels(test_loader,
                                                                                             network)
        all_outputs_train, all_labels_train = metric_learning_utils.get_all_outputs_and_labels(train_loader, network)

        print("Evaluation on train")
        test.full_test_for_representation(k=params.k_for_recall,
                                          all_outputs=all_outputs_train, all_labels=all_labels_train)
        print("Evaluation on test")
        test.full_test_for_representation(k=params.k_for_recall,
                                          all_outputs=all_outputs_test, all_labels=all_labels_test)


def visual_similarity_learning(network, train_loader, test_loader):
    ##################################################################
    #
    # Training for visual similarity
    #
    ##################################################################

    print('Restoring representation network before similarity training')
    if params.recover_representation_net_before_similarity:
        representation_network = utils.load_network_from_checkpoint(network=network,
                                                                    epoch=params.default_recovery_epoch_for_representation,
                                                                    name_prefix_for_saved_model=
                                                                    params.name_prefix_for_saved_model_for_representation)
        representation_length = next(representation_network.fc.modules()).fc.out_features
        print('representation_length = ', representation_length)
        all_outputs_train, all_labels_train = \
            metric_learning_utils.get_all_outputs_and_labels(train_loader, representation_network)
        print('all_outputs_train ', all_outputs_train)
        all_outputs_test, all_labels_test = \
            metric_learning_utils.get_all_outputs_and_labels(test_loader, representation_network)
    else:
        representation_network = None
        representation_length = 256
        all_outputs_train, all_labels_train = spoc.read_spocs_and_labels('all_spocs_file_train_after_pca',
                                                                         'all_labels_file_train')
        all_outputs_test, all_labels_test = spoc.read_spocs_and_labels('all_spocs_file_test_after_pca',
                                                                       'all_labels_file_test')

        #all_outputs_train = torch.from_numpy(np.eye(256)).float().cuda()
        #print('artificial all_outputs_train', all_outputs_train )

    # print('representation_network = ', representation_network)
    print('representation_length = ', representation_length)
    similarity_learning_network = similarity_network_effective.EffectiveSimilarityNetwork(
        number_of_input_features=representation_length, l1_initialization=False).cuda()

    optimizer_for_similarity_learning = optim.Adam(similarity_learning_network.parameters(),
                                                   lr=params.learning_rate_for_similarity)
    # momentum=params.momentum_for_similarity)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_for_similarity_learning,
                                           step_size=params.learning_rate_decay_epoch,
                                           gamma=params.learning_rate_decay_coefficient_for_similarity)

    cosine_similarity_matrix = metric_learning_utils.get_distance_matrix(all_outputs_train,
                                              all_outputs_train,
                                              distance_type=params.distance_type)
    signs_matrix = metric_learning_utils.get_signs_matrix(all_labels_train,
                                              all_labels_train)

    print('cosine_similarity_matrix constant ', cosine_similarity_matrix)
    if params.learn_stage_1:
        # *********
        # Stage 1
        # *********
        metric_learning.metric_learning(all_outputs_train, all_labels_train,
                                        representation_network,
                                        similarity_network=similarity_learning_network,
                                        start_epoch=0,
                                        optimizer=optimizer_for_similarity_learning,
                                        lr_scheduler=exp_lr_scheduler,
                                        criterion=nn.MSELoss(),
                                        stage=1,
                                        all_outputs_test=all_outputs_test, all_labels_test=all_labels_test,
                                        cosine_similarity_matrix=cosine_similarity_matrix,
                                        signs_matrix=None
                                        )
    print('Recover similarity network before the 2 stage')
    similarity_learning_network = utils.load_network_from_checkpoint(network=similarity_learning_network,
                                                                     epoch=params.default_recovery_epoch_for_similarity,
                                                                     name_prefix_for_saved_model=
                                                                     params.name_prefix_for_similarity_saved_model)

    # *********
    # Stage 2
    # *********
    metric_learning.metric_learning(all_outputs_train, all_labels_train,
                                    representation_network,
                                    similarity_network=similarity_learning_network,
                                    start_epoch=0,
                                    optimizer=optimizer_for_similarity_learning,
                                    lr_scheduler=exp_lr_scheduler,
                                    criterion=nn.L1Loss(),
                                    stage=2,
                                    all_outputs_test=all_outputs_test, all_labels_test=all_labels_test,
                                    cosine_similarity_matrix=cosine_similarity_matrix,
                                    signs_matrix=signs_matrix
                                    )


def main():
    pr = cProfile.Profile()
    pr.enable()

    ##################################################################
    #
    # Loading data images
    #
    ##################################################################
    print('Loading data ' + params.dataset)

    if params.dataset == 'cifar':
        train_loader_for_classification, test_loader_for_classification = cifar.download_CIFAR100_for_classification()
        train_loader, test_loader = cifar.download_CIFAR100_for_representation()

    if params.dataset == 'birds':
        train_loader_for_classification, test_loader_for_classification = \
            birds.download_BIRDS_for_classification(data_folder='CUB_200_2011')
        train_loader, test_loader = birds.download_BIRDS_for_representation(data_folder='CUB_200_2011')

    # debug_images_show(train_loader_for_classification)

    ##################################################################
    #
    # Create a network for classification pretrainig and representations learning
    #
    ##################################################################

    print('Create a network ' + params.network)
    if params.network == 'small-resnet':
        network = small_resnet_for_cifar.small_resnet_for_cifar(num_classes=params.num_classes, n=3).cuda()
    if params.network == 'resnet-50':
        network = models.resnet50(pretrained=True).cuda()

        num_ftrs = network.fc.in_features
        network.fc = torch.nn.Sequential()
        network.fc.add_module('fc', nn.Linear(num_ftrs, params.num_classes))
        network.fc.add_module('l2normalization',
                              small_resnet_for_cifar.L2Normalization())  # need normalization for histogramm loss
        network = network.cuda()
        print(network)

    ##################################################################
    #
    # Do classification pretrainig and representations learning
    #
    ##################################################################
    classification_pretrainig(network, train_loader_for_classification, test_loader_for_classification)

    representations_learning(network, train_loader, test_loader)

    ##################################################################
    #
    # Learn visual similarity
    #
    ##################################################################
    visual_similarity_learning(network, train_loader, test_loader)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == '__main__':
    main()
