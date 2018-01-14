from torch.autograd import Variable
import torch
from sklearn.neighbors import NearestNeighbors
import datetime
import numpy as np
import metric_learning_utils
import gc


def test_for_classification(test_loader, network):
    correct = 0
    total = 0
    print('test_loader ', test_loader)
    for data in test_loader:
        images, labels = data
        # print('labels ', labels)

        outputs = network(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        # print('total = ', total)
    accuracy = (100 * correct / total)

    print('Accuracy of the network on the ', total, ' images: %d %%' % accuracy)
    return accuracy


def fraction_of_correct_labels_in_array(actual_label, array, labels):
    correct_labels = len(np.where(labels.cpu().numpy()[array] == actual_label)[0])
    return float(correct_labels) / float(array.shape[0])


def get_total_fraction_of_correct_labels_and_total_number_of_batches(labels, neighbors_lists, number_of_outputs,
                                                                     total_fraction_of_correct_labels,
                                                                     total_number_of_batches):
    total_fraction_of_correct_labels_in_the_batch = 0
    for i in range(number_of_outputs):
        actual_label = labels[i]

        fraction_of_correct_labels_among_the_k_nearest_neighbors = \
            fraction_of_correct_labels_in_array(actual_label, neighbors_lists[i], labels)

        total_fraction_of_correct_labels_in_the_batch = total_fraction_of_correct_labels_in_the_batch + \
                                                        fraction_of_correct_labels_among_the_k_nearest_neighbors
    total_number_of_batches = total_number_of_batches + 1
    fraction_for_this_batch = total_fraction_of_correct_labels_in_the_batch / float(number_of_outputs)
    total_fraction_of_correct_labels = total_fraction_of_correct_labels + fraction_for_this_batch

    return total_fraction_of_correct_labels, total_number_of_batches


def fill_the_distances_matrix(distances_for_pairs, n):
    distances_matrix = np.zeros((n, n))
    start = 0
    for i in range(n - 1):
        distances_matrix[i + 1:, i] = distances_for_pairs[start: start + n - i - 1].data.cpu().numpy()
        distances_matrix[i, i + 1:] = distances_matrix[i + 1:, i]
        distances_matrix[i, i] = np.inf
        start = start + n - i - 1
    # print('distances_matrix ')
    # print(distances_matrix)
    return distances_matrix


def add_distances_matrix_i(distances_matrix_i, distances_matrix, i):
    distances_matrix[i] = distances_matrix_i
    distances_matrix[:, i] = distances_matrix_i
    return distances_matrix


def fill_the_distances_matrix_i(distances_for_pairs, i):
    distances_matrix_i = distances_for_pairs.data.cpu().numpy()
    distances_matrix_i[i] = np.inf
    return distances_matrix_i


def get_neighbors_lists_from_distances_matrix(distances_matrix, k):
    n = distances_matrix.shape[0]
    neighbors_lists = []
    for i in range(n):
        neighbors_for_i = np.argsort(distances_matrix[i])[:k]
        # print('i = ', i, ' neighbors_for_i = ', neighbors_for_i)
        neighbors_lists.append(neighbors_for_i)
    # print('neighbors_lists ', neighbors_lists)
    return neighbors_lists


def get_neighbors_lists(k, labels, number_of_outputs, outputs, similarity_network):
    if similarity_network is None:
        ##########################
        # For representation test we simply compute the distances whilie nearest neighbors search
        ##########################
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(outputs.data.cpu().numpy())
        # these are the lists of indices (inside the current batch) of the k nearest neighbors,
        # not the neighbors vectors themselves
        neighbors_lists = neigh.kneighbors(outputs.data.cpu().numpy(), return_distance=False)
    else:
        ##########################
        # If we have learned visual similarity distances we should find nearest neighbors in another way
        ##########################
        distances_matrix = np.zeros((number_of_outputs, number_of_outputs))
        for i in range(number_of_outputs):
            representation_pairs_i = metric_learning_utils.create_a_batch_of_pairs_i(outputs, i)

            # distances_for_pairs - real distances between the representation vectors
            # distances_for_pairs_from_similarity_network - learned visual similarity distances
            number_of_pairs = representation_pairs_i.data.shape[0]
            #print('number_of_pairs ', number_of_pairs)
            distances_for_pairs_from_similarity_network_i = similarity_network(representation_pairs_i).view(number_of_pairs)

            distances_matrix_i = fill_the_distances_matrix_i(distances_for_pairs_from_similarity_network_i, i)
            distances_matrix = add_distances_matrix_i(distances_matrix_i, distances_matrix, i)

        neighbors_lists = get_neighbors_lists_from_distances_matrix(distances_matrix, k)
        gc.collect()
    return neighbors_lists


# currently is not in use, but I want to save it for some time
def test_for_representation(test_loader, network, k, similarity_network=None):
    total_fraction_of_correct_labels = 0
    total_number_of_batches = 0
    for data in test_loader:
        images, labels = data
        outputs = network(Variable(images).cuda())
        number_of_outputs = outputs.data.shape[0]

        neighbors_lists = get_neighbors_lists(k, labels, number_of_outputs, outputs, similarity_network)

        # here we add new values for current batch to the given
        # total_fraction_of_correct_labels and total_number_of_batches
        total_fraction_of_correct_labels, total_number_of_batches = \
            get_total_fraction_of_correct_labels_and_total_number_of_batches(labels,
                                                                             neighbors_lists,
                                                                             number_of_outputs,
                                                                             total_fraction_of_correct_labels,
                                                                             total_number_of_batches)

    recall_at_k = float(total_fraction_of_correct_labels) / float(total_number_of_batches)
    print('recall_at_', k, ' of the network on the ', total_number_of_batches, ' batches: %f ' % recall_at_k)

    return recall_at_k


def full_test_for_representation(k, all_outputs, all_labels, similarity_network=None):
    total_fraction_of_correct_labels = 0
    total_number_of_batches = 0

    number_of_outputs = all_outputs.shape[0]

    neighbors_lists = get_neighbors_lists(k, all_labels, number_of_outputs, Variable(all_outputs), similarity_network)

    # here we add new values for current batch to the given
    # total_fraction_of_correct_labels and total_number_of_batches
    total_fraction_of_correct_labels, total_number_of_batches = \
        get_total_fraction_of_correct_labels_and_total_number_of_batches(all_labels,
                                                                         neighbors_lists,
                                                                         number_of_outputs,
                                                                         total_fraction_of_correct_labels,
                                                                         total_number_of_batches)

    recall_at_k = float(total_fraction_of_correct_labels) / float(total_number_of_batches)
    print('recall_at_', k, ' of the network on the ', total_number_of_batches, ' batches: %f ' % recall_at_k)

    return recall_at_k
