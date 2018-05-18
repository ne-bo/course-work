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


def metric_learning_poincare(train_loader, network, start_epoch, lr_scheduler,
                             stage, test_loader):
    convolution_spoc_part = network.convolution_spoc_part.cuda()
    similarity_network = network.metric_part.cuda()

    # During the first stage we optimize only parameters of the distance small FC-network
    if stage == 1:
        optimizer = torch.optim.Adam(
            similarity_network.parameters(),
            lr=params.learning_rate_for_poincare_stage_1
        )
    # During the second stage we optimize parameters for the whole network
    if stage == 2:
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=params.learning_rate_for_poincare_stage_2
        )

    vis = visdom.Visdom()
    r_loss = []
    r_recall = []
    iterations = []
    epochs = []
    total_iteration = 0

    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))
    recall_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))

    criterion = torch.nn.MSELoss()
    criterion_margin = margin_loss_for_similarity.MarginLossForSimilarity()
    criterion_hist = histogramm_loss_for_similarity.HistogramLossForSimilarity(150)

    for epoch in range(start_epoch,
                       params.number_of_epochs_for_metric_learning):  # loop over the dataset multiple times
        lr_scheduler.step(epoch=epoch)
        print('current_learning_rate =', optimizer.param_groups[0]['lr'], ' ', datetime.datetime.now())
        i = 0

        for data_i in train_loader:
            inputs_i, labels_i = data_i
            for data_j in train_loader:
                inputs_j, labels_j = data_j

                representation_outputs_1 = convolution_spoc_part(Variable(inputs_i.cuda())).data
                representation_outputs_2 = convolution_spoc_part(Variable(inputs_j.cuda())).data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # pass into similarity network 2 concatenated batches which can be different
                # print('representation_outputs_1 ', representation_outputs_1)
                # print('representation_outputs_2 ', representation_outputs_2)
                similarity_outputs = similarity_network(Variable(torch.cat((representation_outputs_1,
                                                                            representation_outputs_2), dim=0)))

                assert stage in [1, 2], "Stage should be 1 or 2 but actual value is %d " % stage

                distance_matrix_effective = metric_learning_utils.get_distance_matrix(
                    representation_outputs_1,
                    representation_outputs_2,
                    distance_type='poincare'
                )

                # During the first stage we just try to learn poincare distances themselves
                if stage == 1:
                    # use different batches to get all combinations
                    loss = criterion(similarity_outputs.view(-1, 1),
                                     Variable(distance_matrix_effective.contiguous().view(-1, 1)))

                # During the second stage we introduce a margin delta
                # and we add delta to the distance for positive pairs (with the same labels)
                # and subtract the delta from the distance for negative pairs (with the different labels)
                if stage == 2:
                    if params.loss_for_similarity == 'margin':
                        signs_for_pairs = metric_learning_utils.get_signs_matrix(labels_i, labels_j)
                        loss = criterion_margin(similarity_outputs.view(-1, 1),
                                                Variable(signs_for_pairs.contiguous().view(-1, 1)))

                    if params.loss_for_similarity == 'histogramm':
                        signs_for_pairs = metric_learning_utils.get_signs_matrix_for_histogramm_loss(labels_i,
                                                                                                     labels_j)
                        loss = criterion_hist(similarity_outputs.view(params.batch_size_for_similarity,
                                                                      params.batch_size_for_similarity),
                                              Variable(signs_for_pairs))

                loss.backward()
                optimizer.step()

        # print statistics
        if i == 0:
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
            recall_at_k = test.recall_test_for_representation(k=params.k_for_recall, all_outputs=all_outputs_train,
                                                              all_labels=all_labels_train,
                                                              similarity_network=similarity_network)
            r_recall.append(recall_at_k)
            options = dict(legend=['recall for stage ' + str(stage)])
            recall_plot = vis.line(Y=np.array(r_recall), X=np.array(epochs),
                                   # , update='append',
                                   win=recall_plot, opts=options)

            print('Evaluation on test internal')
            recall_at_k = test.recall_test_for_representation(k=params.k_for_recall, all_outputs=all_outputs_test,
                                                              all_labels=all_labels_test,
                                                              similarity_network=similarity_network)

            if stage == 1:
                loss_function_name = ''
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
