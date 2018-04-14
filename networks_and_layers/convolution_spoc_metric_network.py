import torch
import torchvision.models as models
from torch import nn as nn

from networks_and_layers.similarity_network_effective import EffectiveSimilarityNetwork
from networks_and_layers.spoc_layer import Spoc
from utils.utils import get_layer_with_number


class ConvolutionSpocMetric(nn.Module):
    def __init__(self, desired_dimension_for_spoc, initial_PCA_matrix, initial_singular_values,
                 layer_number_in_convolutional_part=11,
                 network_for_convolutional_part=models.alexnet(pretrained=True),
                 number_of_output_neurons=1):
        super(ConvolutionSpocMetric, self).__init__()
        self.desired_dimension_for_spoc = desired_dimension_for_spoc
        self.number_of_output_neurons = number_of_output_neurons

        self.convolution_part = get_layer_with_number(
            network_for_convolutional_part,
            layer_number=layer_number_in_convolutional_part
        )
        print('self.convolution_part ', self.convolution_part)
        self.spoc = Spoc(self.desired_dimension_for_spoc, initial_PCA_matrix, initial_singular_values)
        self.metric_part = EffectiveSimilarityNetwork(
            number_of_input_features=self.desired_dimension_for_spoc,
            number_of_output_neurons=self.number_of_output_neurons
        )

    def forward(self, input):
        output = self.convolution_part(input)
        # print('output after convolution part ', output)
        output = self.spoc(output)
        # print('output after spoc part ', output)
        # here input is a batch of pairs of concatenated spocs
        output = torch.cat((output, output), dim=0)
        # print('output after concatenation the batch with itself ', output)
        output = self.metric_part(output)
        return output

    def __repr__(self):
        return self.__class__.__name__
