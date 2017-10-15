import torch.nn.modules.loss as loss
from torch.nn import functional as F
import numpy as np
import torch
from torch.autograd import Variable


# https://arxiv.org/pdf/1706.07567.pdf
# Sampling Matters in Deep Embedding Learning


class MarginLoss(loss._Loss):
    """Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    D_ij = euclidean distance between representations x_i and x_j
    y_ij = 1 if x_i and x_j represent the same object
    y_ij = -1 otherwise

    margin(i, j) := (alpha + y_ij (D_ij − betha))+
    {loss}(x, y)  = (1/n) * sum_ij (margin(i, j))
.

    `x` and `y` 2D Tensor of size `(minibatch, n)`

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to False, the losses are instead summed for
           each minibatch. Default: True

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, alpha=1.5, betha=0.3):
        super(MarginLoss, self).__init__()
        self.alpha = alpha
        self.bethe = betha

    @staticmethod
    # returns a vector of pairwise distances between
    # the image number i and all images from
    # i + 1 till n
    def get_distances_i(i, input, n, representation_vector_length):  # 0.559    0.000   14.655    0.000
        pdist = torch.nn.PairwiseDistance(p=2).cuda()
        return pdist(Variable(torch.ones([n - i - 1,
                                          representation_vector_length]).cuda()) *
                     input[i].cuda(),
                     input[i + 1:].cuda())

    @staticmethod
    def get_signs_i(i, target, n):
        # here distance wil be 0 for equal targets and non-zero for non equal
        distances_for_signs = Variable(torch.ones([n - i - 1]).cuda()) * target[i].float().cuda() - \
                              target[i + 1:].float().cuda()
        # we should map 0 --> 1 and non-zeero --> -1
        list_of_signs = list(map(lambda d: 1.0 if d == 0.0 else -1.0, distances_for_signs.data))
        return Variable(torch.from_numpy(np.array(list_of_signs, ndmin=2)).float().cuda())

    @staticmethod
    def get_result_i(self, i, distances_i, target, result, n):  # 72.891    0.002  339.828    0.011
        m = distances_i.data.shape[0]
        for j in range(m):
            if target.data[i] != target.data[i + 1 + j]:
                y = -1.0
            else:
                y = 1.0

            result = result + torch.clamp(self.alpha + y * (distances_i[j] - self.bethe), min=0.0).cuda()

      #todo correct this in a right way, then replace the cycle for j with this
      #  signs = self.get_signs_i(i, target, n)
      #  print('distances', distances_i)
      #  print('Variable(torch.ones([m]).cuda()) * self.bethe', Variable(torch.ones([m, 1]).fill_(self.bethe).cuda()))
      #  distances_i_bethe = distances_i - Variable(torch.ones([m, 1]).fill_(self.bethe).cuda())
      #  print('distances_i_bethe', distances_i_bethe)
      #  print('signs', signs)
      #  print('signs * distances_i_bethe', torch.transpose(distances_i_bethe, 0, 1) * torch.transpose(signs, 0, 1))
      #  result = result + torch.clamp(self.alpha + distances_i_bethe * signs, min=0.0).cuda()

        return result

    def forward(self, input, target):
        loss._assert_no_grad(target)

        n = input.data.shape[0]
        representation_vector_length = input[0].data.shape[0]

        result = 0.0
        for i in range(n - 1):
            distances_i = self.get_distances_i(i, input, n, representation_vector_length)
            result = self.get_result_i(self, i, distances_i, target, result, n)

        if self.size_average:
            result = torch.mean(result).cuda()
        return result
