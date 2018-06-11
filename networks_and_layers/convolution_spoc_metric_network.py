import torch
from torch import nn as nn

from networks_and_layers.similarity_network_effective import EffectiveSimilarityNetwork


class ConvolutionSpocMetric(nn.Module):
    def __init__(self, convolution_spoc_part,
                 number_of_output_neurons=1):
        super(ConvolutionSpocMetric, self).__init__()
        self.convolution_spoc_part = convolution_spoc_part.cuda()
        self.number_of_output_neurons = number_of_output_neurons
        self.metric_part = EffectiveSimilarityNetwork(
            number_of_input_features=self.convolution_spoc_part.desired_dimension_for_spoc,
            number_of_output_neurons=self.number_of_output_neurons).cuda()

    def forward(self, input):
        output = self.convolution_spoc_part(input)
        # here input is a batch of pairs of concatenated spocs
        output = torch.cat((output, output), dim=0)
        # print('output after concatenation the batch with itself ', output)
        output = self.metric_part(output)
        return output

    def __repr__(self):
        return self.__class__.__name__
