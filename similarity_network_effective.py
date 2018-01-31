import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter


class AllPairs(nn.Module):
    def __init__(self, in_features, out_features):
        super(AllPairs, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(self.in_features, self.out_features).cuda()
        self.fc2 = nn.Linear(self.in_features, self.out_features).cuda()

    def forward(self, input):
        #print('input', input)
        # split the input to 2 parts corresponding to 2 different batches
        batch_size = input.size(0)//2
        input_1 = self.fc1(input[:batch_size])
        input_2 = self.fc2(input[batch_size:])

        #print('input_1 ', input_1)
        #print('input_2 ', input_2)
        input_1 = input_1.expand(input_1.size(0), input_1.size(0), input_1.size(1))
        #print('input_1 after the expansion ', input_1)
        input_2 = torch.transpose(input_2.expand(input_2.size(0), input_2.size(0), input_2.size(1)), 0, 1)
        #print('input_2 after the expansion and transposition', input_2)

        return input_1 + input_2

    def __repr__(self):
        return self.__class__.__name__


class EffectiveSimilarityNetwork(nn.Module):
    def __init__(self, number_of_input_features):
        super(EffectiveSimilarityNetwork, self).__init__()

        ##################################################################
        #
        # Network parameters
        #
        ##################################################################

        # Learning Non-Metric Visual Similarity for Image Retrieval
        # https://arxiv.org/abs/1709.01353
        # Here we have architecture B from the original article

        self.number_of_input_features = number_of_input_features

        # parameters for first fully connected layer
        self.number_of_hidden_neurons_for_1_fully_connected = 2048  # 4096

        # parameters for second fully connected layer
        self.number_of_hidden_neurons_for_2_fully_connected = 2048  # 4096

        self.number_of_output_neurons = 1

        ##################################################################
        #
        # Network structure
        #
        ##################################################################
        self.fc1 = AllPairs(in_features=self.number_of_input_features,
                            out_features=self.number_of_hidden_neurons_for_1_fully_connected)

        self.fc2 = nn.Linear(self.number_of_hidden_neurons_for_1_fully_connected,
                             self.number_of_hidden_neurons_for_2_fully_connected)

        self.fc3 = nn.Linear(self.number_of_hidden_neurons_for_2_fully_connected,
                             self.number_of_output_neurons)

        ##################################################################
        # Weights initialization
        ##################################################################

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #print('x after the all pairs layer', x)
        x = F.relu(self.fc2(x))
        #reg = torch.nn.Dropout(p=0.5)
       # y = x
       # print('x after the first linear layer', x, ' ', y.sum())
        x = self.fc3(x)
        #print('x ', x.view(300, 300))
        return x

# def create_similarity_network(number_of_input_features):
#    return SimilarityNetwork(number_of_input_features)
