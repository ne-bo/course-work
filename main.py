import cifar
import learning
import test
import net
import params
import torchvision
import utils
import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import loss
import natasha_resnet
from torch.optim import lr_scheduler
import small_resnet_for_cifar
import cProfile
import pstats
import io


def main():
    pr = cProfile.Profile()
    pr.enable()

    print('Loading data')

    train_loader, test_loader, \
    train_loader_for_classification, test_loader_for_classification = cifar.download_CIFAR100()

    ##################################################################
    #
    # Images show for debug
    #
    ##################################################################
    # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # show images
    # print("images.shape ", images.shape)
    # utils.imshow(torchvision.utils.make_grid(images)) #images = Tensor of shape (B x C x H x W)
    # print labels
    # print(' '.join('%5s' % labels[j] for j in range(params.batch_size)))

    print('Create a network')
    network = small_resnet_for_cifar.small_resnet_for_cifar(num_classes=params.num_classes, n=3).cuda()

    restore_epoch = 0
    optimizer = optim.SGD(network.parameters(),
                          lr=params.learning_rate,
                          momentum=params.momentum)
    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    if params.recover_classification:
        print('Restore for classification pre-training')
        restore_epoch = params.defalut_recovery_epoch_for_classification
        network, optimizer = utils.load_network_and_optimizer_from_checkpoint(network=network,
                                                                              optimizer=optimizer,
                                                                              epoch=restore_epoch)

    print('Create a multi_lr_scheduler')
    # Decay LR by a factor of 0.1 every 10 epochs
    multi_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[82, 123],
                                                  gamma=0.1)

    ##################################################################
    #
    # Pre-training for classification
    #
    ##################################################################
    if params.learn_classification:
        print('Start classification pretraining')
        learning.learning_process(train_loader=train_loader_for_classification,
                                  network=network,
                                  criterion=nn.CrossEntropyLoss(),
                                  test_loader=test_loader_for_classification,
                                  mode=params.mode_classification,
                                  optimizer=optimizer,
                                  start_epoch=restore_epoch,
                                  lr_scheduler=multi_lr_scheduler)

    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    if params.recover_classification_net_before_representation:
        print('Restoring before representational training')
        network = utils.load_network_from_checkpoint(network=network,
                                                     epoch=160)

    ##################################################################
    #
    # Training for representations with our margin loss
    #
    ##################################################################

    print('Representational training')

    optimizer_for_representational_learning = optim.SGD(network.parameters(),
                                                        lr=params.learning_rate,
                                                        momentum=params.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                           step_size=params.learning_rate_decay_epoch,
                                           gamma=params.learning_rate_decay_coefficient)

    learning.learning_process(train_loader=train_loader,
                              network=network,
                              criterion=loss.MarginLoss(),
                              test_loader=test_loader,
                              mode=params.mode_representation,
                              optimizer=optimizer_for_representational_learning,
                              lr_scheduler=exp_lr_scheduler)

    print("Evaluation: ")

    test.test_for_representation(test_loader=train_loader,
                                 network=network,
                                 k=params.k_for_recall)
    test.test_for_representation(test_loader=test_loader,
                                 network=network,
                                 k=params.k_for_recall)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == '__main__':
    main()
