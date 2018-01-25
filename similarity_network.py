import math
import torch.nn as nn
import torch.nn.functional as F


class SimilarityNetwork(nn.Module):
    def __init__(self, number_of_input_features):
        super(SimilarityNetwork, self).__init__()

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
        self.fc1 = nn.Linear(self.number_of_input_features,
                             self.number_of_hidden_neurons_for_1_fully_connected)

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
