import gc
import torchvision.models as models

from metric_learning_utils import get_distance_matrix
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
import cProfile
import pstats
import io


def cos_dist(x, y):
    xy = np.dot(x, y);
    xx = np.dot(x, x);
    yy = np.dot(y, y);

    return -xy * 1.0 / np.sqrt(xx * yy)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = torch.sqrt(x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)))
    return dist


def metric_learning(all_outputs_train, all_labels_train,
                    representation_network, similarity_network,
                    start_epoch,
                    optimizer,
                    lr_scheduler,
                    criterion, stage,
                    all_outputs_test, all_labels_test):
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

        number_of_batches = all_outputs_train.shape[0] // params.batch_size_for_similarity
        print('number_of_batches = ', number_of_batches)

        # todo try to use really all possible pairs, not only pairs within the stable batches
        for i in np.random.permutation(range(number_of_batches)):
            for j in np.random.permutation(range(number_of_batches)):
                representation_outputs_1 = all_outputs_train[
                                           i * params.batch_size_for_similarity:
                                           (i + 1) * params.batch_size_for_similarity]
                labels_1 = all_labels_train[
                           i * params.batch_size_for_similarity: (i + 1) * params.batch_size_for_similarity]
                representation_outputs_2 = all_outputs_train[
                                           j * params.batch_size_for_similarity:
                                           (j + 1) * params.batch_size_for_similarity]
                labels_2 = all_labels_train[
                           j * params.batch_size_for_similarity: (j + 1) * params.batch_size_for_similarity]

                #print('representation_outputs_1 ', representation_outputs_1)
                #print('representation_outputs_2 ', representation_outputs_2)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # pass into similarity network 2 concatenated batches which can be different
                similarity_outputs = similarity_network(Variable(torch.cat((representation_outputs_1,
                                                                            representation_outputs_2), dim=0)))

                #print('weights', similarity_network.fc1.fc1.weight)

                # Learning Non-Metric Visual Similarity for Image Retrieval
                # https://arxiv.org/abs/1709.01353

                assert stage in [1, 2, 3], "Stage should be 1, 2 or 3 but actual value is %d " % stage

                # During the first stage we just try to learn distances themselves
                if stage == 1:
                    # use different batches to get all combinations
                    distance_matrix_effective = get_distance_matrix(representation_outputs_1,
                                                                    representation_outputs_2,
                                                                    distance_type='euclidean')
                    # print('similarity_outputs', similarity_outputs)
                    # we use tril here because our distance matrix is symmetric
                    #print('similarity_outputs ', similarity_outputs.view(params.batch_size_for_similarity,
                    #                                                    params.batch_size_for_similarity))
                    #print('distance_matrix_effective', distance_matrix_effective.view(params.batch_size_for_similarity,
                    #                                                         params.batch_size_for_similarity))
                    #input()
                    loss = criterion((similarity_outputs.view(params.batch_size_for_similarity *
                                                                        params.batch_size_for_similarity, -1)),
                                     Variable(distance_matrix_effective.view(params.batch_size_for_similarity *
                                                                             params.batch_size_for_similarity, -1)))
                    #print('loss = ', loss)
                    #input()
                # During the second stage we introduce a margin delta
                # and we add delta to the distance for positive pairs (with the same labels)
                # and subtract the delta from the distance for negative pairs (with the different labels)
                if stage == 2:
                    # pair_distances_with_deltas = distances_for_pairs + params.delta_for_similarity * signs_for_pairs
                    # loss = criterion(similarity_outputs, pair_distances_with_deltas)
                    # todo fill signs and cosine similarities matrices
                    cosine_similarities = None
                    signs_for_pairs = None
                    cosine_similarities_with_deltas = cosine_similarities + params.delta_for_similarity * signs_for_pairs
                    loss = criterion(similarity_outputs, cosine_similarities_with_deltas)
                # todo add hard examples for stage 3

                loss.backward()
                optimizer.step()

        # print statistics
        current_batch_loss = loss.data[0]
        print('[ephoch %d, itteration in the epoch %5d] loss: %.30f' %
          (epoch + 1, i + 1, current_batch_loss))
        #print('similarity_network.fc1.fc1.weight = ', similarity_network.fc1.fc1.weight)
        #print('similarity_network.fc1.fc2.weight = ', similarity_network.fc1.fc2.weight)
        #print('fc2.weight = ', similarity_network.fc2.weight)
        print('fc3.weight = ', similarity_network.fc3.weight)
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
            gc.collect()

            print('Evaluation on train\n')
            recall_at_k = test.partial_test_for_representation(k=params.k_for_recall,
                                                               all_outputs=all_outputs_train,
                                                               all_labels=all_labels_train,
                                                               similarity_network=similarity_network)

            print('Evaluation on test\n')
            # todo return this evaluation after speed up the calculations
            # recall_at_k = test.full_test_for_representation(k=params.k_for_recall,
            #                                                all_outputs=all_outputs_test, all_labels=all_labels_test,
            #                                                similarity_network=similarity_network)

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


