import cProfile
import datetime

import numpy as np
import visdom
from torch.autograd import Variable

import utils
from evaluation import test
from utils import metric_learning_utils, params, utils


def learning_process(train_loader, network, criterion,
                     test_loader, mode, optimizer=None, start_epoch=0,
                     lr_scheduler=None):
    vis = visdom.Visdom()
    r_loss = []
    iterations = []
    total_iteration = 0

    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))

    number_of_epochs = 0
    name_prefix_for_saved_model = ''
    if mode == params.mode_classification:
        number_of_epochs = params.number_of_epochs_for_classification
        name_prefix_for_saved_model = params.name_prefix_for_saved_model_for_classification
    if mode == params.mode_representation:
        number_of_epochs = params.number_of_epochs_for_representation
        name_prefix_for_saved_model = params.name_prefix_for_saved_model_for_representation

    for epoch in range(start_epoch, number_of_epochs):  # loop over the dataset multiple times
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
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            current_batch_loss = loss.data[0]
            if i % params.skip_step == 0:  # print every 2000 mini-batches
                print('[epoch %d, iteration in the epoch %5d] loss: %.30f' % (epoch + 1, i + 1, current_batch_loss))

                r_loss.append(current_batch_loss)
                iterations.append(total_iteration + i)

                options = dict(legend=['loss for' + mode])
                loss_plot = vis.line(Y=np.array(r_loss), X=np.array(iterations), win=loss_plot, opts=options)

        if epoch % 10 == 0:
            # print the train accuracy at every epoch
            # to see if it is enough to start representation training
            # or we should proceed with classification
            if mode == params.mode_classification:
                accuracy = test.test_for_classification(test_loader=test_loader,
                                                        network=network)
            if mode == params.mode_representation:
                # we should recalculate all outputs before the evaluation because our network changed during the trainig
                all_outputs_test, all_labels_test = metric_learning_utils.get_all_outputs_and_labels(test_loader,
                                                                                                     network)
                recall_at_k = test.recall_test_for_representation(
                    k=params.k_for_recall,
                    all_outputs=all_outputs_test,
                    all_labels=all_labels_test
                )
            utils.save_checkpoint(
                network=network,
                optimizer=optimizer,
                filename=name_prefix_for_saved_model + '-%d' % epoch,
                epoch=epoch
            )
        total_iteration = total_iteration + i
        print('total_iteration = ', total_iteration)

    print('Finished Training')
