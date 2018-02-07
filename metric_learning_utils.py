import numpy as np
import torch
from torch.autograd import Variable

import loss
import histogramm_loss


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


def cos_dist(x, y):
    xy = torch.dot(x, y)
    xx = torch.dot(x, x)
    yy = torch.dot(y, y)
    #print('xy', xy)
    #print('xx', xx)
    #print('yy', yy)
    return xy/np.sqrt(xx * yy)

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def get_distance_matrix(representation_outputs_1, representation_outputs_2, distance_type='euclidean'):
    n = representation_outputs_1.size(0)
    d = representation_outputs_1.size(1)

    x = representation_outputs_1.unsqueeze(1).expand(n, n, d)
    y = representation_outputs_2.unsqueeze(0).expand(n, n, d)
    # print('x ', x)
    # print('y ', y)
    if distance_type == 'euclidean':
        return torch.sqrt(torch.pow(x - y, 2).sum(2))
    else:
        if distance_type == 'l1':
            return torch.abs(x - y).sum(2)
        else:
            if distance_type == 'cosine':  # todo actually this code doesn't give us cosime distances need to correct it
                cosine_similarity_matrix = sklearn_cosine_similarity(representation_outputs_1.cpu().numpy(),
                                                                     representation_outputs_2.cpu().numpy())
                #print('cosine_similarity_matrix ', cosine_similarity_matrix)
                return torch.from_numpy(cosine_similarity_matrix).float().cuda()
            else:
                raise Exception('You should use euclidean, l1, cosine distance or histogram loss!')


def myfunc(a):
    # we should map 0 --> 1 and non-zero --> -1
    if a == 0.0:
        return 1.0
    else:
        return -1.0


def get_signs_matrix(labels1, labels2):
    distances_between_labels = get_distance_matrix(labels1.unsqueeze(1).float(),
                                                   labels2.unsqueeze(1).float(),
                                                   distance_type='euclidean')
    print('distances_between_labels ', distances_between_labels)
    # we should map 0 --> 1 and non-zero --> -1
    vfunc = np.vectorize(myfunc)
    signs = torch.from_numpy(vfunc(distances_between_labels.cpu().numpy())).float().cuda()
    print('signs ', signs)
    return signs
