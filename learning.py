import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import params
import utils
import torch
from loss import MarginLoss
import test
import visdom
import numpy as np
import datetime
from torch.optim import lr_scheduler
import cProfile
import pstats
import io
import cifar


def learning_process(train_loader,
                     network,
                     criterion,
                     test_loader,
                     mode,
                     optimizer=None,
                     start_epoch=0,
                     lr_scheduler=lr_scheduler):
    vis = visdom.Visdom()
    r_loss = []
    iterations = []
    total_iteration = 0

    loss_plot = vis.line(Y=np.zeros(1), X=np.zeros(1))

    for epoch in range(start_epoch, params.number_of_epochs):  # loop over the dataset multiple times
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
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)

            # representation = network.get_representation(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            current_batch_loss = loss.data[0]
            if i % params.skip_step == 0:  # print every 2000 mini-batches
                print('[ephoch %d, itteration in the epoch %5d] loss: %.30f' %
                      (epoch + 1, i + 1, current_batch_loss))

                print('conv1 = ', network.conv1.weight[0][0])

                r_loss.append(current_batch_loss)
                iterations.append(total_iteration + i)

                options = dict(legend=['loss for' + mode])
                loss_plot = vis.line(Y=np.array(r_loss), X=np.array(iterations),
                                     # , update='append',
                                     win=loss_plot, opts=options)

                # print the train accuracy at every epoch
                # to see if it is enough to start representation training
                # or we should proceed with classification
                if mode == params.mode_classification:
                    accuracy = test.test_for_classification(test_loader=test_loader,
                                                            network=network)
                if mode == params.mode_representation:
                    recall_at_k = test.test_for_representation(test_loader=test_loader,
                                                               network=network,
                                                               k=params.k_for_recall)

        if epoch % 10 == 0:
            utils.save_checkpoint(network=network,
                                  optimizer=optimizer,
                                  filename=params.name_prefix_for_saved_model + '-%d' % epoch,
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

    print('Finished Training')
