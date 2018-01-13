import torchvision.models as models

from metric_learning_utils import create_a_batch_of_pairs
from small_resnet_for_cifar import L2Normalization
import birds
from torch.autograd import Variable
import torch.nn as nn
import params
import utils
import torch
import test
import visdom
import numpy as np
import datetime
from torch.optim import lr_scheduler
import cProfile
import pstats
import io


def metric_learning(train_loader, test_loader,
                    representation_network, similarity_network,
                    start_epoch,
                    optimizer,
                    lr_scheduler,
                    criterion, stage):
    vis = visdom.Visdom()
    r_loss = []
    iterations = []
    total_iteration = 0

    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))

    for epoch in range(start_epoch,
                       params.number_of_epochs_for_metric_learning):  # loop over the dataset multiple times
        pr = cProfile.Profile()
        pr.enable()

        lr_scheduler.step(epoch=epoch)
        print('current_learning_rate =', optimizer.param_groups[0]['lr'])
        print(datetime.datetime.now())
        running_loss = 0.0
        i = 0

        # for representation we need clever sampling which should change every epoch
        # if mode == params.mode_representation:
        #    train_loader, test_loader, \
        #    train_loader_for_classification, test_loader_for_classification = cifar.download_CIFAR100()

        for i, data in enumerate(train_loader, 0):
            # print('i = ', i)
            # get the inputs
            # inputs are [torch.FloatTensor of size 4x3x32x32]
            # labels are [torch.LongTensor of size 4]
            # here 4 is a batch size and 3 is a number of channels in the input images
            # 32x32 is a size of input image
            initial_images, labels = data

            # here we take representations andd create a batch of pairs
            representation_outputs = representation_network(Variable(initial_images).cuda())
            representation_pairs, \
            distances_for_pairs, \
            signs_for_pairs = create_a_batch_of_pairs(representation_outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            similarity_outputs = similarity_network(representation_pairs)

            # Learning Non-Metric Visual Similarity for Image Retrieval
            # https://arxiv.org/abs/1709.01353

            assert stage in [1, 2, 3], "Stage should be 1, 2 or 3 but actual value is %d " % stage

            # During the first stage we just try to learn distances themselves
            if stage == 1:
                loss = criterion(similarity_outputs, distances_for_pairs)
            # During the second stage we introduce a margin delta
            # and we add delta to the distance for positive pairs (with the same labels)
            # and subtract the delta from the distance for negative pairs (with the different labels)
            if stage == 2:
                pair_distances_with_deltas = distances_for_pairs + params.delta_for_similarity * signs_for_pairs
                loss = criterion(similarity_outputs, pair_distances_with_deltas)
            # todo add hard examples for stage 3

            loss.backward()
            optimizer.step()

            # print statistics
            current_batch_loss = loss.data[0]
            if i % params.skip_step == 0:  # print every 2000 mini-batches
                print('[ephoch %d, itteration in the epoch %5d] loss: %.30f' %
                      (epoch + 1, i + 1, current_batch_loss))

                r_loss.append(current_batch_loss)
                iterations.append(total_iteration + i)

                options = dict(legend=['loss for stage ' + str(stage)])
                loss_plot = vis.line(Y=np.array(r_loss), X=np.array(iterations),
                                     # , update='append',
                                     win=loss_plot, opts=options)


        if epoch % 10 == 0:
            # print the quality metric
            # Here evaluation is heavy so we do it only every 10 epochs
            print('similarity_network ', similarity_network)
            recall_at_k = test.full_test_for_representation(test_loader=test_loader,
                                                            network=representation_network,
                                                            k=params.k_for_recall,
                                                            similarity_network=similarity_network)

            utils.save_checkpoint(network=similarity_network,
                                  optimizer=optimizer,
                                  filename=params.name_prefix_for_similarity_saved_model + '-%d' % epoch,
                                  epoch=epoch)
        total_iteration = total_iteration + i
        print('total_iteration = ', total_iteration)

        pr.disable()
        # s = io.FileIO('profiler-statistic')
        s = io.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

    print('Finished Training for similarity learning for stage %d ' % stage)


def test_of_generating_batch_of_pairs():
    network = models.resnet50(pretrained=True).cuda()

    num_ftrs = network.fc.in_features
    network.fc = torch.nn.Sequential()
    network.fc.add_module('fc', nn.Linear(num_ftrs, params.num_classes))
    network.fc.add_module('l2normalization', L2Normalization())  # need normalization for histogramm loss
    network = network.cuda()
    print(network)

    train_loader, test_loader = birds.download_BIRDS_for_representation(data_folder='CUB_200_2011')

    for i, data in enumerate(train_loader, 0):
        # print('i = ', i)
        # get the inputs
        # inputs are [torch.FloatTensor of size 4x3x32x32]
        # labels are [torch.LongTensor of size 4]
        # here 4 is a batch size and 3 is a number of channels in the input images
        # 32x32 is a size of input image
        initial_images, labels = data

        # here we take representations andd create a batch of pairs
        representation_outputs = network(Variable(initial_images).cuda())
        representation_pairs, \
        distances_for_pairs, \
        signs_for_pairs = create_a_batch_of_pairs(representation_outputs, labels)

# test_of_generating_batch_of_pairs()
