import net
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import params


def learning_process(train_loader, network):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=params.learning_rate, momentum=params.momentum)

    for epoch in range(params.number_of_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)

            #print(inputs)
            #print(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % params.skip_step == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / params.skip_step))
                running_loss = 0.0

    print('Finished Training')

