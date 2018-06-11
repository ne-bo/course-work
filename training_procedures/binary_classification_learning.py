import datetime
import gc

import numpy as np
import visdom
from torch.autograd import Variable

from evaluation import test
from utils import params, utils
from utils.metric_learning_utils import get_indices_for_loss


def binary_learning(train_loader,
                    network,
                    criterion,
                    test_loader,
                    optimizer,
                    start_epoch,
                    lr_scheduler):
    vis = visdom.Visdom()
    r_loss = []
    r_average_f1 = []
    iterations = []
    epochs = []
    total_iteration = 0

    options = dict(legend=['loss'])
    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1), opts=options)
    options = dict(legend=['average_f1'])
    average_f1_plot = vis.line(Y=np.zeros(1), X=np.zeros(1), opts=options)

    for epoch in range(start_epoch, params.number_of_epochs_for_metric_learning):

        print('current_learning_rate =', optimizer.param_groups[0]['lr'], ' ', datetime.datetime.now())

        i = 0
        for data in train_loader:
            i = i + 1
            inputs, labels = data
            # print('inputs ', inputs) # batch_size x 3 x 64 x 64
            # we need pairs of images in our batch
            # print('inputs, labels ', labels)
            # and +1/-1 labels matrix

            labels_matrix = utils.get_labels_matrix_fast(labels, labels).view(-1, 1)

            indices_for_loss = get_indices_for_loss(labels_matrix, negative_pair_sign=0)

            labels_matrix = labels_matrix[indices_for_loss]
            labels_matrix = Variable(labels_matrix).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # here we should create input pair for the network from just inputs
            outputs = network(Variable(inputs).cuda())

            outputs = outputs[indices_for_loss.cuda(), :]
            # print('outputs ', outputs)
            # print('labels_matrix.long().view(-1, 1).squeeze() ', labels_matrix.long().view(-1, 1).squeeze())
            loss = criterion(outputs, labels_matrix.long().view(-1, 1).squeeze())

            loss.backward()
            optimizer.step()

            # print statistics
            current_batch_loss = loss.data[0]

            if i % 10 == 0:  # print every 2000 mini-batches
                print('[epoch %d, iteration in the epoch %5d] loss: %.30f' % (epoch + 1, i + 1, current_batch_loss))
                # print('PCA matrix ', network.spoc.PCA_matrix)

                r_loss.append(current_batch_loss)
                iterations.append(total_iteration + i)

                options = dict(legend=['loss'])
                loss_plot = vis.line(Y=np.array(r_loss), X=np.array(iterations), win=loss_plot, opts=options)

        lr_scheduler.step(epoch=epoch, metrics=current_batch_loss)

        if epoch % 10 == 0:
            epochs.append(epoch)
            # print the quality metric
            gc.collect()

            print('Evaluation on train internal', datetime.datetime.now())
            average_f1 = test.test_for_binary_classification_1_batch(train_loader, network)
            r_average_f1.append(average_f1)
            options = dict(legend=['average_f1'])
            average_f1_plot = vis.line(Y=np.array(r_average_f1), X=np.array(epochs), win=average_f1_plot, opts=options)

            print('Evaluation on test internal', datetime.datetime.now())
            average_f1 = test.test_for_binary_classification_1_batch(test_loader, network)

            utils.save_checkpoint(
                network=network,
                optimizer=optimizer,
                filename=params.name_prefix_for_saved_model_for_binary_classification + '-%d' % (epoch),
                epoch=epoch
            )
        total_iteration = total_iteration + i

    print('Finished Training for binary classification')


