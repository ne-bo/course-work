import torch
from torch.autograd import Variable
import numpy as np
import torchtricks.criterion


# Learning Deep Embeddings with Histogram Loss
# https://arxiv.org/abs/1611.00822
import params


class MarginLossForSimilarity(torch.nn.Module):
    def __init__(self, alpha=0.3, bethe=1.2):
        super(MarginLossForSimilarity, self).__init__()
        self.alpha = alpha
        self.bethe = bethe
        print('self.alpha = ', self.alpha)
        print('self.bethe = ', self.bethe)

    def forward(self, distances_matrix, signs_matrix):
        """
        D_ij =  euclidean distance between representations x_i and x_j
        y_ij =  1 if x_i and x_j represent the same object
        y_ij = -1 otherwise

        margin(i, j) := (alpha + y_ij (D_ij − bethe))+
        {loss}        = (1/n) * sum_ij (margin(i, j))

        """
        n = float(distances_matrix.data.shape[0])
        #print('inputed distances matrix ', distances_matrix)
        #distances_matrix = distances_matrix.view(params.batch_size_for_similarity, params.batch_size_for_similarity)
        #print('after reshape distances matrix ', distances_matrix)
        margin = torch.clamp(self.alpha + signs_matrix * (distances_matrix - self.bethe), min=0.0).cuda()
        loss = torch.sum(margin)/n
        return loss
