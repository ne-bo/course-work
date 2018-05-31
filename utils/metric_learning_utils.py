import gc
import sys
from os import path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity, \
    euclidean_distances as sklearn_euclidean_distances, manhattan_distances as sklearn_l1_distances
from torch.autograd import Variable

# https://github.com/facebookresearch/poincare-embeddings
sys.path.append(path.abspath('/home/natasha/PycharmProjects/poincare-embeddings/'))
from model import PoincareDistance


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


def get_all_outputs_and_labels_for_large_dataset(test_loader, network):
    all_outputs = torch.cuda.FloatTensor()
    all_labels = torch.LongTensor()
    i = 0
    pack_volume = 10
    print('len(test_loader) ', len(test_loader.dataset.train_images))
    dataset_length = len(test_loader.dataset.train_images)
    for data in test_loader:
        images, labels = data
        outputs = network(Variable(images).cuda())
        all_outputs = torch.cat((all_outputs, outputs.data), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
        if i % pack_volume == 0:
            torch.save(all_outputs, '/tmp/all_outputs_%d' % i)
            torch.save(all_labels, '/tmp/all_labels_%d' % i)
            print('saving i = ', i)
            all_outputs = torch.cuda.FloatTensor()
            all_labels = torch.LongTensor()
            gc.collect()
        i = i + 1
    number_of_batches = i
    print('number_of_batches ', number_of_batches)
    all_outputs = torch.FloatTensor()
    all_labels = torch.LongTensor()
    for i in range(number_of_batches):
        if i % pack_volume == 0:
            outputs = torch.load('/tmp/all_outputs_%d' % i).cpu()
            labels = torch.load('/tmp/all_labels_%d' % i)
            all_outputs = torch.cat((all_outputs, outputs), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            print('reading i = ', i, 'outputs ', outputs.shape)

            gc.collect()

    return all_outputs, all_labels


def get_distance_matrix(representation_outputs_1, representation_outputs_2, distance_type='euclidean'):

    allowed_distances_types = ['euclidean', 'l1', 'cosine', 'poincare']
    assert distance_type in allowed_distances_types, \
        "Distance type should be  in %s but actual value is %s " % (allowed_distances_types, distance_type)

    x = representation_outputs_1.cpu().numpy()
    y = representation_outputs_2.cpu().numpy()

    if distance_type == 'euclidean':
        matrix = sklearn_euclidean_distances(x, y)

    if distance_type == 'l1':
        matrix = sklearn_l1_distances(x, y)

    if distance_type == 'cosine':
        matrix = sklearn_cosine_similarity(x, y)

    if distance_type == 'poincare':
        distfn = PoincareDistance
        matrix = np.zeros((x.shape[0], y.shape[0]))
        for i, u in enumerate(x):
            for j, v in enumerate(y):
                dist_ij = distfn()(Variable(torch.from_numpy(u)), Variable(torch.from_numpy(v)))
                matrix[i, j] = dist_ij.data.cpu().numpy()

    return torch.from_numpy(matrix).float().cuda()


def myfunc(a):
    # we should map 0 --> 1 and non-zero --> -1
    if a == 0.0:
        return 1.0
    else:
        return -1.0


def myfunc_for_histogram_loss(a):
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


def get_signs_matrix_for_histogram_loss(labels1, labels2):
    distances_between_labels = get_distance_matrix(labels1.unsqueeze(1).float(),
                                                   labels2.unsqueeze(1).float(),
                                                   distance_type='euclidean')
    # print('distances_between_labels ', distances_between_labels)
    # we should map 0 --> 1 and non-zero --> 0
    vfunc = np.vectorize(myfunc_for_histogram_loss)
    signs = torch.from_numpy(vfunc(distances_between_labels.cpu().numpy())).byte().cuda()
    # print('signs ', signs)
    return signs


def get_indices_for_loss(labels_matrix, negative_pair_sign=0):
    if 1 in labels_matrix.cpu().numpy():
        indices_of_positive_pairs = torch.from_numpy(np.where(labels_matrix.cpu().numpy() == 1)[0])
        indices_of_negative_pairs = torch.from_numpy(np.where(labels_matrix.cpu().numpy() == negative_pair_sign)[0])
        number_of_positive_pairs = indices_of_positive_pairs.shape[0]
        number_of_negative_pairs = indices_of_negative_pairs.shape[0]
        # print('number_of_positive_pairs ', number_of_positive_pairs)
        # print('number_of_negative_pairs ', number_of_negative_pairs)
        # print('indices_of_positive_pairs ', indices_of_positive_pairs)
        if number_of_negative_pairs >= number_of_positive_pairs:
            permutation = torch.randperm(number_of_negative_pairs)
            indices_of_negative_pairs = indices_of_negative_pairs[permutation]
            indices_of_negative_pairs = indices_of_negative_pairs[:number_of_positive_pairs]
        indices_for_loss = torch.cat((indices_of_positive_pairs, indices_of_negative_pairs), dim=0)
        # print('indices_for_loss ', indices_for_loss)
    else:
        indices_for_loss = None
    return indices_for_loss
