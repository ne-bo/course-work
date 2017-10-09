import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        ##################################################################
        #
        # Network parameters
        #
        ##################################################################
        self.num_classes = num_classes

        # parameters for first convolution layer
        self.input_channels = 3
        self.number_of_filters_for_1_convolution = 6
        self.filter_size_for_1_convolution = 5

        # parameters for 1 pooling
        self.pooling_scale_for_1_pooling = 2
        self.pooling_stride_for_1_pooling = 2

        # parameters for second convolution layer
        self.number_of_filters_for_2_convolution = 16
        self.filter_size_for_2_convolution = 5

        # parameters for first fully connected layer
        self.number_of_hidden_neurons_for_1_fully_connected = 120

        # parameters for second fully connected layer
        self.number_of_hidden_neurons_for_2_fully_connected = 84

        ##################################################################
        #
        # Network structure
        #
        ##################################################################
        self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                               out_channels=self.number_of_filters_for_1_convolution,
                               kernel_size=self.filter_size_for_1_convolution,
                               stride=1,
                               padding=0,
                               bias=True)

        self.pool = nn.MaxPool2d(kernel_size=self.pooling_scale_for_1_pooling,
                                 stride=self.pooling_stride_for_1_pooling,
                                 padding=0)

        self.conv2 = nn.Conv2d(in_channels=self.number_of_filters_for_1_convolution,
                               out_channels=self.number_of_filters_for_2_convolution,
                               kernel_size=self.filter_size_for_2_convolution,
                               stride=1,
                               padding=0,
                               bias=True)
        self.fc1 = nn.Linear(self.number_of_filters_for_2_convolution *
                             self.filter_size_for_2_convolution *
                             self.filter_size_for_2_convolution,
                             self.number_of_hidden_neurons_for_1_fully_connected)

        self.fc2 = nn.Linear(self.number_of_hidden_neurons_for_1_fully_connected,
                             self.number_of_hidden_neurons_for_2_fully_connected)
        # here the last layer has name fc instead of fc3 to be easy replaceable with resnet18 architecture
        self.fc = nn.Linear(self.number_of_hidden_neurons_for_2_fully_connected,
                            num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,
                   self.number_of_filters_for_2_convolution *
                   self.filter_size_for_2_convolution *
                   self.filter_size_for_2_convolution)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

    # representation is
    # [torch.cuda.FloatTensor of size
    # batch_size x
    # number_of_filters_for_2_convolution x
    # filter_size_for_2_convolution x
    # filter_size_for_2_convolution
    #  (GPU 0)]
    def get_representation(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

