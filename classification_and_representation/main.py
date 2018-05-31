import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo
import torchvision
import torchvision.models as models
from torch.optim import lr_scheduler

import networks_and_layers.l2_normalization
import utils
from classification_and_representation import spoc
from datasets.loaders import birds, cifar
from evaluation import test
from losses import histogram_loss
from networks_and_layers import similarity_network_effective, small_resnet_for_cifar
from training_procedures import learning, metric_learning
from utils import metric_learning_utils, params, utils


def debug_images_show(loader):
    ##################################################################
    #
    # Images show for debug
    #
    ##################################################################
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()
    # show images
    print("images.shape ", images.shape)
    utils.imshow(torchvision.utils.make_grid(images))  # images = Tensor of shape (B x C x H x W)
    # print labels
    # todo python 3.6 has so-called f-strings. This is like in Groovy when you can insert the value directly into print
    print(' '.join('%5s' % labels[j] for j in range(params.batch_size)))


def classification_pretrainig(network, train_loader_for_classification, test_loader_for_classification):
    restore_epoch = params.default_recovery_epoch_for_classification
    optimizer = optim.SGD(
        network.parameters(),
        lr=params.learning_rate_for_classification,
        momentum=params.momentum_for_classification
    )
    # todo Extract methods is a useful refactoring especially for code parts with such big comments
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    if params.recover_classification:
        print('Restore for classification pre-training')
        restore_epoch = params.default_recovery_epoch_for_classification
        network, optimizer = utils.load_network_and_optimizer_from_checkpoint(
            network=network,
            optimizer=optimizer,
            epoch=restore_epoch,
            name_prefix_for_saved_model=params.name_prefix_for_saved_model_for_classification
        )
        start_epoch = restore_epoch
    else:
        start_epoch = 0

    print('Create a multi_lr_scheduler')
    # Decay LR by a factor of 0.1 every 10 epochs
    multi_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[82, 123],
        gamma=0.1
    )

    ##################################################################
    #
    # Classification for pre-training
    #
    ##################################################################
    if params.learn_classification:
        print('Start classification pretraining')
        learning.learning_process(train_loader=train_loader_for_classification, network=network,
                                  criterion=nn.CrossEntropyLoss(), test_loader=test_loader_for_classification,
                                  mode=params.mode_classification, optimizer=optimizer, start_epoch=start_epoch,
                                  lr_scheduler=multi_lr_scheduler)


def representations_learning(network, train_loader, test_loader):
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    # todo look carefully in control flow ifs, there are too many of them and they duplicate each other
    if params.recover_classification_net_before_representation:
        print('Restoring before representational training')
        restore_epoch = params.default_recovery_epoch_for_classification
        network = utils.load_network_from_checkpoint(
            network=network,
            epoch=restore_epoch,
            name_prefix_for_saved_model=
            params.name_prefix_for_saved_model_for_classification
        )

    ##################################################################
    #
    # Training for representations with our margin loss
    #
    ##################################################################

    print('Representational training')

    optimizer_for_representational_learning = optim.Adam(
        network.parameters(),
        lr=params.learning_rate_for_representation
    )  # ,
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
        learning.learning_process(train_loader=train_loader, network=network,
                                  criterion=histogram_loss.HistogramLoss(150), test_loader=test_loader,
                                  mode=params.mode_representation, optimizer=optimizer_for_representational_learning,
                                  start_epoch=start_epoch, lr_scheduler=exp_lr_scheduler)

        print("Evaluation: ")
        all_outputs_test, all_labels_test = metric_learning_utils.get_all_outputs_and_labels(test_loader,
                                                                                             network)
        all_outputs_train, all_labels_train = metric_learning_utils.get_all_outputs_and_labels(train_loader, network)

        print("Evaluation on train")
        test.recall_test_for_representation(k=params.k_for_recall, all_outputs=all_outputs_train,
                                            all_labels=all_labels_train)
        print("Evaluation on test")
        test.recall_test_for_representation(k=params.k_for_recall,
                                            all_outputs=all_outputs_test, all_labels=all_labels_test)


def visual_similarity_learning(network, train_loader, test_loader):
    ##################################################################
    #
    # Training for visual similarity
    #
    ##################################################################

    print('Restoring representation network before similarity training')
    if params.recover_representation_net_before_similarity:
        representation_network = utils.load_network_from_checkpoint(
            network=network,
            epoch=params.default_recovery_epoch_for_representation,
            name_prefix_for_saved_model=
            params.name_prefix_for_saved_model_for_representation
        )

        representation_length = next(representation_network.fc.modules()).fc.out_features
        print('representation_length = ', representation_length)
        all_outputs_train, all_labels_train = \
            metric_learning_utils.get_all_outputs_and_labels(train_loader, representation_network)
        print('all_outputs_train ', all_outputs_train)
        all_outputs_test, all_labels_test = \
            metric_learning_utils.get_all_outputs_and_labels(test_loader, representation_network)
    else:
        print('Wee take outputs and labels directly from the files')
        representation_network = None
        representation_length = 256
        all_outputs_train, all_labels_train = spoc.read_spocs_and_labels('all_spocs_file_train_after_pca',
                                                                         'all_labels_file_train')
        all_outputs_test, all_labels_test = spoc.read_spocs_and_labels('all_spocs_file_test_after_pca',
                                                                       'all_labels_file_test')

    print('representation_length = ', representation_length)
    similarity_learning_network = similarity_network_effective.EffectiveSimilarityNetwork(
        number_of_input_features=representation_length,
        l1_initialization=False,
        add_dropout=True
    ).cuda()

    # print('Recover similarity network before the 1 stage')
    # similarity_learning_network = utils.load_network_from_checkpoint(network=similarity_learning_network,
    #                                                              epoch=params.default_recovery_epoch_for_similarity,
    #                                                                name_prefix_for_saved_model=
    #                                                                params.name_prefix_for_similarity_saved_model,
    #                                                                 stage=1)

    optimizer_for_similarity_learning = optim.Adam(similarity_learning_network.parameters(),
                                                   lr=params.learning_rate_for_similarity)
    # momentum=params.momentum_for_similarity)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_for_similarity_learning,
                                           step_size=params.learning_rate_decay_epoch,
                                           gamma=params.learning_rate_decay_coefficient_for_similarity)

    if params.learn_stage_1:
        # *********
        # Stage 1
        # *********
        metric_learning.metric_learning(all_outputs_train, all_labels_train,
                                        similarity_network=similarity_learning_network, start_epoch=0,
                                        optimizer=optimizer_for_similarity_learning, lr_scheduler=exp_lr_scheduler,
                                        criterion=nn.MSELoss(), stage=1,
                                        all_outputs_test=all_outputs_test, all_labels_test=all_labels_test)

    print('Recover similarity network before the 2 stage')
    similarity_learning_network = utils.load_network_from_checkpoint(
        network=similarity_learning_network,
        epoch=params.default_recovery_epoch_for_similarity,
        name_prefix_for_saved_model=params.name_prefix_for_similarity_saved_model,
        stage=1
    )
    # for epoch in range(0, 100, 10):
    #    print('Recover similarity network before the 2 stage ---- epoch %d', epoch)
    #    similarity_learning_network = utils.load_network_from_checkpoint(network=similarity_learning_network,
    #                                                                 epoch=epoch,
    #                                                                 name_prefix_for_saved_model=
    #                                                                 params.name_prefix_for_similarity_saved_model,
    #                                                                 stage=1)

    print('Evaluation on train after the stage 1')
    recall_at_k = test.recall_test_for_representation(
        k=params.k_for_recall,
        all_outputs=all_outputs_train,
        all_labels=all_labels_train,
        similarity_network=similarity_learning_network
    )
    print('Evaluation on test after the stage 1')
    recall_at_k = test.recall_test_for_representation(
        k=params.k_for_recall,
        all_outputs=all_outputs_test,
        all_labels=all_labels_test,
        similarity_network=similarity_learning_network
    )

    # *********
    # Stage 2
    # *********
    if params.learn_stage_2:
        metric_learning.metric_learning(all_outputs_train, all_labels_train,
                                        similarity_network=similarity_learning_network, start_epoch=0,
                                        optimizer=optimizer_for_similarity_learning, lr_scheduler=exp_lr_scheduler,
                                        criterion=nn.MSELoss(), stage=2, all_outputs_test=all_outputs_test,
                                        all_labels_test=all_labels_test)
    print('Recover similarity network after the 2 stage')
    similarity_learning_network = utils.load_network_from_checkpoint(network=similarity_learning_network,
                                                                     epoch=params.default_recovery_epoch_for_similarity,
                                                                     name_prefix_for_saved_model=
                                                                     params.name_prefix_for_similarity_saved_model,
                                                                     stage=2,
                                                                     loss_function_name=params.loss_for_similarity)
    print('Evaluation on train after the stage 2')
    recall_at_k = test.recall_test_for_representation(k=params.k_for_recall, all_outputs=all_outputs_train,
                                                      all_labels=all_labels_train,
                                                      similarity_network=similarity_learning_network)
    print('Evaluation on test after the stage 2')
    recall_at_k = test.recall_test_for_representation(k=params.k_for_recall, all_outputs=all_outputs_test,
                                                      all_labels=all_labels_test,
                                                      similarity_network=similarity_learning_network)


def main():
    ##################################################################
    #
    # Loading data images
    #
    ##################################################################
    print('Loading data ' + params.dataset)
    train_loader, test_loader = None, None
    # todo python has enums, you can use them instead of string like 'cifar'
    if params.dataset == 'cifar' and (params.learn_classification or params.learn_representation):
        train_loader_for_classification, test_loader_for_classification = cifar.download_CIFAR100_for_classification()
        train_loader, test_loader = cifar.download_CIFAR100_for_representation()

    if params.dataset == 'birds' and (params.learn_classification or params.learn_representation):
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
    network = None
    if params.network == 'small-resnet' and (params.learn_classification or params.learn_representation):
        network = small_resnet_for_cifar.small_resnet_for_cifar(num_classes=params.num_classes, n=3).cuda()
    if params.network == 'resnet-50' and (params.learn_classification or params.learn_representation):
        network = models.resnet50(pretrained=True).cuda()

        num_ftrs = network.fc.in_features
        network.fc = torch.nn.Sequential()
        network.fc.add_module('fc', nn.Linear(num_ftrs, params.num_classes))
        network.fc.add_module('l2normalization',
                              networks_and_layers.l2_normalization.L2Normalization())  # need normalization for histogram loss
        network = network.cuda()
        print(network)

    ##################################################################
    #
    # Do classification pretrainig and representations learning
    #
    ##################################################################
    if params.learn_classification:
        classification_pretrainig(network, train_loader_for_classification, test_loader_for_classification)

    if params.learn_representation:
        representations_learning(network, train_loader, test_loader)

    ##################################################################
    #
    # Learn visual similarity
    #
    ##################################################################
    visual_similarity_learning(network, train_loader, test_loader)


if __name__ == '__main__':
    main()
