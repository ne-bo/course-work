import datetime
import gc

import numpy as np
import torch
import visdom
from torch.autograd import Variable

import utils
from evaluation import test
from losses import histogramm_loss_for_similarity, margin_loss_for_similarity
from utils import metric_learning_utils, params


def metric_learning(all_outputs_train, all_labels_train,
                    representation_network, similarity_network,
                    start_epoch,
                    optimizer,
                    lr_scheduler,
                    criterion, stage,
                    all_outputs_test, all_labels_test,
                    cosine_similarity_matrix,
                    signs_matrix
                    ):
    vis = visdom.Visdom()
    r_loss = []
    r_recall = []
    iterations = []
    epochs = []
    total_iteration = 0

    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))
    recall_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))

    criterion_margin = margin_loss_for_similarity.MarginLossForSimilarity()
    criterion_hist = histogramm_loss_for_similarity.HistogramLossForSimilarity(150)

    n = all_outputs_train.shape[0]
    number_of_batches = all_outputs_train.shape[0] // params.batch_size_for_similarity

    # reorder outputs and labels for histogramm loss for UKB!!!!!!!
    if params.sampling_for_similarity:
        all_labels_train, indices = torch.sort(all_labels_train)
        print('all_labels_train sorted ', all_labels_train)
        print('indices ', indices)
        all_outputs_train = all_outputs_train[indices.cuda()]
        print('all_outputs_train sorted ', all_outputs_train)

    cosine_similarity_matrix = metric_learning_utils.get_distance_matrix(all_outputs_train,
                                                                         all_outputs_train,
                                                                         distance_type=params.distance_type)

    if params.loss_for_similarity == 'histogramm':
        signs_matrix = metric_learning_utils.get_signs_matrix_for_histogramm_loss(all_labels_train, all_labels_train)
    else:
        signs_matrix = metric_learning_utils.get_signs_matrix(all_labels_train, all_labels_train)

    print('reordered cosine_similarity_matrix constant ', cosine_similarity_matrix)
    print('reordered  signs_matrix ', signs_matrix)

    for epoch in range(start_epoch,
                       params.number_of_epochs_for_metric_learning):  # loop over the dataset multiple times
        lr_scheduler.step(epoch=epoch)
        print('current_learning_rate =', optimizer.param_groups[0]['lr'], ' ', datetime.datetime.now())
        i = 0
        if params.sampling_for_similarity:
            j_limit = 1
        else:
            j_limit = number_of_batches

        for i in range(number_of_batches):
            for j in range(j_limit):
                if params.sampling_for_similarity:
                    j = i

                representation_outputs_1 = all_outputs_train[
                                           i * params.batch_size_for_similarity:
                                           (i + 1) * params.batch_size_for_similarity]
                representation_outputs_2 = all_outputs_train[
                                           j * params.batch_size_for_similarity:
                                           (j + 1) * params.batch_size_for_similarity]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # pass into similarity network 2 concatenated batches which can be different
                #print('representation_outputs_1 ', representation_outputs_1)
                #print('representation_outputs_2 ', representation_outputs_2)
                similarity_outputs = similarity_network(Variable(torch.cat((representation_outputs_1,
                                                                            representation_outputs_2), dim=0)))

                # Learning Non-Metric Visual Similarity for Image Retrieval
                # https://arxiv.org/abs/1709.01353

                assert stage in [1, 2, 3], "Stage should be 1, 2 or 3 but actual value is %d " % stage

                distance_matrix_effective = cosine_similarity_matrix[i * params.batch_size_for_similarity:
                                                                     (i + 1) * params.batch_size_for_similarity,
                                            j * params.batch_size_for_similarity:
                                            (j + 1) * params.batch_size_for_similarity
                                            ]

                # During the first stage we just try to learn distances themselves
                if stage == 1:
                    # use different batches to get all combinations
                    loss = criterion(similarity_outputs.view(-1, 1),
                                     Variable(distance_matrix_effective.contiguous().view(-1, 1)))
                    #print('similarity_outputs.view(-1, 1)', similarity_outputs.view(params.batch_size_for_similarity,
                    #                                                                params.batch_size_for_similarity))
                    #print('distance_matrix_effective ', distance_matrix_effective)
                    #print('i = ', i, 'j = ', j, ' loss = ', loss.data[0])
                    #input()
                # During the second stage we introduce a margin delta
                # and we add delta to the distance for positive pairs (with the same labels)
                # and subtract the delta from the distance for negative pairs (with the different labels)
                if stage == 2:
                    signs_for_pairs = signs_matrix[i * params.batch_size_for_similarity:
                                                   (i + 1) * params.batch_size_for_similarity,
                                      j * params.batch_size_for_similarity:
                                      (j + 1) * params.batch_size_for_similarity
                                      ]
                    if params.loss_for_similarity == 'delta':
                        cosine_similarities_with_deltas = distance_matrix_effective + \
                                                          params.delta_for_similarity * signs_for_pairs
                        loss = criterion(similarity_outputs.view(-1, 1),
                                     Variable(cosine_similarities_with_deltas.view(-1, 1)))

                    if params.loss_for_similarity == 'margin':
                        loss = criterion_margin(similarity_outputs.view(-1, 1),
                                            Variable(signs_for_pairs.contiguous().view(-1, 1)))

                    if params.loss_for_similarity == 'histogramm':
                        #print('similarity_outputs ', similarity_outputs)
                        loss = criterion_hist(similarity_outputs.view(params.batch_size_for_similarity,
                                                                      params.batch_size_for_similarity),
                                              Variable(signs_for_pairs))

                # todo add hard examples for stage 3

                loss.backward()
                optimizer.step()

                # print statistics
                if i == 0 and j == 0:
                    current_batch_loss = loss.data[0]
                    print('[ephoch %d, itteration in the epoch %5d] loss: %.30f' %
                          (epoch + 1, i + 1, current_batch_loss))
                    r_loss.append(current_batch_loss)

        iterations.append(total_iteration + i)
        options = dict(legend=['loss for stage ' + str(stage)])
        loss_plot = vis.line(Y=np.array(r_loss), X=np.array(iterations),
                             # , update='append',
                             win=loss_plot, opts=options)

        if epoch % 10 == 0:
            epochs.append(epoch)
            # print the quality metric
            # Here evaluation is heavy so we do it only every 10 epochs
            # print('similarity_network ', similarity_network)
            gc.collect()

            print('Evaluation on train internal')
            recall_at_k = test.partial_test_for_representation(k=params.k_for_recall,
                                                               all_outputs=all_outputs_train,
                                                               all_labels=all_labels_train,
                                                               similarity_network=similarity_network)
            r_recall.append(recall_at_k)
            options = dict(legend=['recall for stage ' + str(stage)])
            recall_plot = vis.line(Y=np.array(r_recall), X=np.array(epochs),
                                 # , update='append',
                                 win=recall_plot, opts=options)

            print('Evaluation on test internal')
            recall_at_k = test.partial_test_for_representation(k=params.k_for_recall,
                                                               all_outputs=all_outputs_test, all_labels=all_labels_test,
                                                               similarity_network=similarity_network)

            if stage == 1:
                loss_function_name =''
            else:
                loss_function_name = params.loss_for_similarity
            utils.save_checkpoint(network=similarity_network,
                                  optimizer=optimizer,
                                  filename=params.name_prefix_for_similarity_saved_model + '-%d-%d%s' % (epoch,
                                                                                                         stage,
                                                                                                         loss_function_name),
                                  epoch=epoch)
        total_iteration = total_iteration + number_of_batches
        print('total_iteration = ', total_iteration)


    print('Finished Training for similarity learning for stage %d ' % stage)
