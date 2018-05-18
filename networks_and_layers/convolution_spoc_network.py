import torchvision.models as models
from torch import nn as nn

from networks_and_layers.spoc_layer import Spoc
from utils.utils import get_layer_with_number


class ConvolutionSpoc(nn.Module):
    def __init__(self, desired_dimension_for_spoc, initial_PCA_matrix, initial_singular_values,
                 layer_number_in_convolutional_part=11,
                 network_for_convolutional_part=models.alexnet(pretrained=True)):
        super(ConvolutionSpoc, self).__init__()
        self.desired_dimension_for_spoc = desired_dimension_for_spoc

        self.convolution_part = get_layer_with_number(
            network_for_convolutional_part,
            layer_number=layer_number_in_convolutional_part
        ).cuda()
        print('self.convolution_part ', self.convolution_part)
        self.spoc = Spoc(self.desired_dimension_for_spoc, initial_PCA_matrix, initial_singular_values).cuda()

    def forward(self, input):
        output = self.convolution_part(input)
        # print('output after convolution part ', output)
        output = self.spoc(output)
        return output

    def __repr__(self):
        return self.__class__.__name__
