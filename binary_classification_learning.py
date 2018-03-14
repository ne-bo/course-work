import datetime
import gc

import numpy as np
import torch
import visdom
from torch.autograd import Variable

import params
import utils


def get_labels_matrix(labels_list_1, labels_list_2):
    matrix = torch.from_numpy(np.zeros((labels_list_1.shape[0], labels_list_2.shape[0])))
    for i, labels1 in enumerate(labels_list_1):
        for j, labels2 in enumerate(labels_list_2):
            matrix[i, j] = int(np.in1d(labels1, labels2).any())
    print('matrix ', matrix)
    return matrix


def binary_learning(train_loader,
                    network,
                    criterion,
                    test_loader,
                    optimizer,
                    start_epoch,
                    lr_scheduler):
    vis = visdom.Visdom()
    r_loss = []
    r_recall = []
    iterations = []
    epochs = []
    total_iteration = 0

    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))
    recall_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))

    for epoch in range(start_epoch, params.number_of_epochs_for_metric_learning):
        lr_scheduler.step(epoch=epoch)
        print('current_learning_rate =', optimizer.param_groups[0]['lr'], ' ', datetime.datetime.now())

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # we need pairs of images in our batch
            input_pairs = Variable(torch.cat((inputs.view(params.batch_size_for_binary_classification, 105 * 105, 1),
                                              inputs.view(params.batch_size_for_binary_classification, 105 * 105, 1))))
            # and +1/-1 labels matrix

            labels_matrix =  Variable(get_labels_matrix(labels, labels))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # here we should create input pair for the network from just inputs
            outputs = network(input_pairs)

            loss = criterion(outputs, labels_matrix)

            loss.backward()
            optimizer.step()

            # print statistics
            current_batch_loss = loss.data[0]
            print('[ephoch %d, itteration in the epoch %5d] loss: %.30f' %
                          (epoch + 1, i + 1, current_batch_loss))
            r_loss.append(current_batch_loss)

        iterations.append(total_iteration + i)
        options = dict(legend=['loss'])
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
            recall_at_k = 0.0
            r_recall.append(recall_at_k)
            options = dict(legend=['recall'])
            recall_plot = vis.line(Y=np.array(r_recall), X=np.array(epochs),
                                 # , update='append',
                                 win=recall_plot, opts=options)

            print('Evaluation on test internal')
            recall_at_k = 0.0

            utils.save_checkpoint(network=network,
                                 optimizer=optimizer,
                                  filename=params.name_prefix_for_similarity_saved_model + '-%d' % (epoch),
                                  epoch=epoch)

    print('Finished Training for binary classification')
