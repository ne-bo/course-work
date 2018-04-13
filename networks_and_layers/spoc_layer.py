import torch
from torch import nn as nn
from torch.autograd import Variable

from networks_and_layers.l2_normalization import L2Normalization
from utils.spoc import compute_spoc_by_outputs, learn_PCA_matrix_for_spocs


class Spoc(nn.Module):
    def __init__(self, desired_dimension):
        super(Spoc, self).__init__()
        self.desired_dimension = desired_dimension
        print('desired_dimension =', desired_dimension)

    # input is the output of a some convolutional network
    # shape: batch_size x number_of_channels x convolutional_map_size x convolutional_map_size
    #
    # the shape of output of this layer is batch_size x desired_dimension
    def forward(self, input):
        spocs = compute_spoc_by_outputs(input)

        # print('spocs before pca ', spocs)

        PCA_matrix, singular_values = learn_PCA_matrix_for_spocs(spocs, self.desired_dimension)

        spocs = torch.div(torch.mm(spocs, Variable(PCA_matrix)), Variable(singular_values))

        # print('spocs after pca ', spocs)
        normalization = L2Normalization()
        spocs = normalization(spocs)

        return spocs

    def __repr__(self):
        return self.__class__.__name__