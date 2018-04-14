import torch
from torch import nn as nn

from networks_and_layers.l2_normalization import L2Normalization
from utils.spoc import compute_spoc_by_outputs


class Spoc(nn.Module):
    def __init__(self, desired_dimension, initial_PCA_matrix, initial_singular_values):
        super(Spoc, self).__init__()
        self.desired_dimension = desired_dimension
        print('desired_dimension =', desired_dimension)
        self.PCA_matrix = nn.Parameter(initial_PCA_matrix, requires_grad=True)
        self.singular_values = nn.Parameter(initial_singular_values, requires_grad=True)

    # input is the output of a some convolutional network
    # shape: batch_size x number_of_channels x convolutional_map_size x convolutional_map_size
    #
    # the shape of output of this layer is batch_size x desired_dimension
    def forward(self, input):
        spocs = compute_spoc_by_outputs(input)

        spocs = torch.div(torch.mm(spocs, self.PCA_matrix), self.singular_values)

        # print('spocs after pca ', spocs)
        normalization = L2Normalization()
        spocs = normalization(spocs)

        return spocs

    def __repr__(self):
        return self.__class__.__name__