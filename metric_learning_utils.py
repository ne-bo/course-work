import numpy as np
import torch
from torch.autograd import Variable

import loss


def get_all_outputs_and_labels(test_loader, network):
    all_outputs = torch.cuda.FloatTensor()
    all_labels = torch.LongTensor()
    for data in test_loader:
        images, labels = data
        outputs = network(Variable(images).cuda())
        all_outputs = torch.cat((all_outputs, outputs.data), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
    print('all_outputs', all_outputs)
    print('all_labels', all_labels)
    return all_outputs, all_labels


def get_distance_matrix(representation_outputs_1, representation_outputs_2, distance_type='euclidean'):
    n = representation_outputs_1.size(0)
    d = representation_outputs_1.size(1)

    x = representation_outputs_1.unsqueeze(1).expand(n, n, d)
    y = representation_outputs_2.unsqueeze(0).expand(n, n, d)
    #print('x ', x)
    #print('y ', y)
    if distance_type == 'euclidean':
        return torch.sqrt(torch.pow(x - y, 2).sum(2))
    else:
        if distance_type == 'l1':
            return torch.abs(x - y).sum(2)
        else:
            if distance_type == 'cosine':
                dot_product_11 = torch.dot(representation_outputs_1, representation_outputs_1)
                dot_product_22 = torch.dot(representation_outputs_2, representation_outputs_2)
                dot_product_12 = torch.dot(representation_outputs_1, representation_outputs_2)
                norm_1 = torch.sqrt(dot_product_11 * dot_product_11)
                norm_2 = torch.sqrt(dot_product_22 * dot_product_22)
                return torch.div(-dot_product_12, norm_1 * norm_2)
            else:
                raise Exception('You should use euclidean, l1 or cosine distance!')