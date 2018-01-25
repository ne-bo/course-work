import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter


class AllPairs(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AllPairs, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_1 = Parameter(torch.Tensor(out_features, in_features))
        self.weight_2 = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_1 = Parameter(torch.Tensor(out_features))
            self.bias_2 = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_1', None)
            self.register_parameter('bias_2', None)
        self.reset_parameters()

    @staticmethod
    def reset_weight_and_bias(weight, bias):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        if bias is not None:
            bias.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        self.reset_weight_and_bias(self.weight_1, self.bias_1)
        self.reset_weight_and_bias(self.weight_2, self.bias_2)

    def forward(self, input):
        #print('input', input)
        fc1 = nn.Linear(self.in_features, self.out_features).cuda()
        fc2 = nn.Linear(self.in_features, self.out_features).cuda()
        input_1 = fc1(input)
        input_2 = fc2(input)

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
        self.number_of_hidden_neurons_for_1_fully_connected = 1024  # 4096

        # parameters for second fully connected layer
        self.number_of_hidden_neurons_for_2_fully_connected = 1024  # 4096

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
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# def create_similarity_network(number_of_input_features):
#    return SimilarityNetwork(number_of_input_features)
