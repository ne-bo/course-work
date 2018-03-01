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


from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances as sklearn_euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances as sklearn_l1_distances

def get_distance_matrix(representation_outputs_1, representation_outputs_2, distance_type='euclidean'):
    n = representation_outputs_1.size(0)
    d = representation_outputs_1.size(1)

    x = representation_outputs_1.unsqueeze(1).expand(n, n, d)
    y = representation_outputs_2.unsqueeze(0).expand(n, n, d)
    # print('x ', x)
    # print('y ', y)
    if distance_type == 'euclidean':
        euclidean_distances_matrix = sklearn_euclidean_distances(representation_outputs_1.cpu().numpy(),
                                                                 representation_outputs_2.cpu().numpy())
        return torch.from_numpy(euclidean_distances_matrix).float().cuda()

        # return torch.sqrt(torch.pow(x - y, 2).sum(2))
    else:
        if distance_type == 'l1':
            l1_distances_matrix = sklearn_l1_distances(representation_outputs_1.cpu().numpy(),
                                                                     representation_outputs_2.cpu().numpy())
            return torch.from_numpy(l1_distances_matrix).float().cuda()
            #return torch.abs(x - y).sum(2)
        else:
            if distance_type == 'cosine':
                cosine_similarity_matrix = sklearn_cosine_similarity(representation_outputs_1.cpu().numpy(),
                                                                     representation_outputs_2.cpu().numpy())
                # print('cosine_similarity_matrix ', cosine_similarity_matrix)
                return torch.from_numpy(cosine_similarity_matrix).float().cuda()
            else:
                raise Exception('You should use euclidean, l1, cosine distance or histogram loss!')


def myfunc(a):
    # we should map 0 --> 1 and non-zero --> -1
    if a == 0.0:
        return 1.0
    else:
        return -1.0


def myfunc_for_histogramm_loss(a):
    # we should map 0 --> 1 and non-zero --> 0
    if a == 0.0:
        return 1
    else:
        return 0


def get_signs_matrix(labels1, labels2):
    distances_between_labels = get_distance_matrix(labels1.unsqueeze(1).float(),
                                                   labels2.unsqueeze(1).float(),
                                                   distance_type='euclidean')
    # print('distances_between_labels ', distances_between_labels)
    # we should map 0 --> 1 and non-zero --> -1
    vfunc = np.vectorize(myfunc)
    signs = torch.from_numpy(vfunc(distances_between_labels.cpu().numpy())).float().cuda()
    # print('signs ', signs)
    return signs


def get_signs_matrix_for_histogramm_loss(labels1, labels2):
    distances_between_labels = get_distance_matrix(labels1.unsqueeze(1).float(),
                                                   labels2.unsqueeze(1).float(),
                                                   distance_type='euclidean')
    # print('distances_between_labels ', distances_between_labels)
    # we should map 0 --> 1 and non-zero --> 0
    vfunc = np.vectorize(myfunc_for_histogramm_loss)
    signs = torch.from_numpy(vfunc(distances_between_labels.cpu().numpy())).byte().cuda()
    # print('signs ', signs)
    return signs
