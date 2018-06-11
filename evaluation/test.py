import gc

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.autograd import Variable

from utils import metric_learning_utils, params, utils


#################################################
#
# Vanilla classification accuracy test
#
#################################################
def test_for_classification(test_loader, network):
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data

        outputs = network(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    accuracy = (100 * correct / total)

    print('Accuracy of the network on the ', total, ' images: %d %%' % accuracy)
    return accuracy


#################################################
#
# f1 test for binary classification of positive/negative pairs
# NOT for all possible pairs
#
#################################################
def test_for_binary_classification_1_batch(test_loader, network):
    f1 = 0.0
    number_of_small_matrices = 0.0
    ground_truth_sum = 0
    predicted_sum = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        # and +1/-1 labels matrix
        labels_matrix = Variable(utils.get_labels_matrix_fast(labels, labels)).cuda()

        # here we should create input pair for the network from just inputs
        outputs = network(Variable(inputs).cuda()).data
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()

        ground_truth = labels_matrix.long().view(-1, 1).squeeze().data.cpu().numpy()

        ground_truth_sum = ground_truth_sum + np.sum(ground_truth)
        predicted_sum = predicted_sum + np.sum(predicted)
        total = total + ground_truth.shape[0]

        f1 = f1 + f1_score(ground_truth, predicted)
        number_of_small_matrices = number_of_small_matrices + 1.0

    average_f1 = f1 / number_of_small_matrices
    print('average f1 = ', average_f1)
    print('ground_truth_sum/total =', ground_truth_sum / total)
    print('predicted_sum/total =', predicted_sum / total)
    return average_f1


#################################################
#
# Recall test for retrieval results
# For all possible pairs
#
#################################################
def recall_test_for_representation(k, all_outputs, all_labels, similarity_network=None, several_labels=False):
    distances_matrix = compute_the_distance_matrix_with_network(all_outputs, similarity_network, params.distance_type)

    print('number of 0 label examples  ', np.where(all_labels.cpu().numpy() == 0)[0].shape)
    nearest_neighbors = get_nearest_neighbors_from_distances_matrix(distances_matrix, k,
                                                                    distance_type=params.distance_type)

    recall_at_k = get_recall(all_labels, nearest_neighbors, several_labels)
    print('recall_at_', k, ' of the network: %f ' % recall_at_k)

    return recall_at_k


def MAP_test_for_representation(k, all_outputs, all_labels, similarity_network=None):
    distances_matrix = compute_the_distance_matrix_with_network(all_outputs, similarity_network, params.distance_type)

    print('number of 0 label examples  ', np.where(all_labels.cpu().numpy() == 0)[0].shape)
    nearest_neighbors = get_nearest_neighbors_from_distances_matrix(distances_matrix, k,
                                                                    distance_type=params.distance_type)

    sign_matrix = utils.get_labels_matrix_fast(all_labels, all_labels)
    MAP_at_k = get_MAP_at_k(sign_matrix, nearest_neighbors)
    print('MAP_at_', k, ' of the network: %f ' % MAP_at_k)

    return MAP_at_k


#################################################
#
# Helper functions
#
#################################################

def compute_the_distance_matrix_with_network(all_outputs, similarity_network, distance_type='euclidean'):
    number_of_outputs = all_outputs.shape[0]
    number_of_batches = all_outputs.shape[0] // params.batch_size_for_similarity
    print('number_of_batches = ', number_of_batches)

    distances_matrix = torch.from_numpy(np.zeros((number_of_outputs, number_of_outputs))).float()
    for i in range(number_of_batches):
        for j in range(number_of_batches):
            representation_outputs_1 = all_outputs[
                                       i * params.batch_size_for_similarity:(i + 1) * params.batch_size_for_similarity]
            representation_outputs_2 = all_outputs[
                                       j * params.batch_size_for_similarity:(j + 1) * params.batch_size_for_similarity]
            if similarity_network is not None:
                similarity_outputs = similarity_network(Variable(torch.cat(
                    (representation_outputs_1,
                     # NATASHA!!!!!!!!!!!!!!!!!  here interchanging representation_outputs_1 and representation_outputs_2
                     # fixes results for unordered outputs!!!!
                     representation_outputs_2), dim=0))).view(params.batch_size_for_similarity,
                                                              params.batch_size_for_similarity)
                # Or transpose here can also fix
                similarity_outputs = torch.transpose(similarity_outputs, 0, 1)
            else:
                similarity_outputs = Variable(metric_learning_utils.get_distance_matrix(representation_outputs_1,
                                                                                        representation_outputs_2,
                                                                                        distance_type=distance_type))
            distances_matrix[
            i * params.batch_size_for_similarity:(i + 1) * params.batch_size_for_similarity,
            j * params.batch_size_for_similarity:(j + 1) * params.batch_size_for_similarity] = similarity_outputs.data
            gc.collect()

    return distances_matrix


# Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
def get_recall(all_labels, nearest_neighbors, several_labels=False):
    if several_labels:
        return get_recall_for_several_labels(all_labels, nearest_neighbors)
    recall_at_k = 0.0
    all_labels = all_labels.cpu().numpy()
    for i, current_label in enumerate(all_labels):
        total_number_of_this_label_in_the_dataset = np.count_nonzero(all_labels == current_label)
        indices_of_nearest_neighbors = np.asarray(nearest_neighbors[i], dtype=int)
        labels_of_nearest_neighbors = all_labels[indices_of_nearest_neighbors]
        number_of_this_label_among_k_nearest_neighbors = np.count_nonzero(labels_of_nearest_neighbors == current_label)
        recall_at_k = \
            recall_at_k + number_of_this_label_among_k_nearest_neighbors / total_number_of_this_label_in_the_dataset

    recall_at_k = recall_at_k / (float(all_labels.shape[0]))

    return recall_at_k


# Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
def get_recall_for_several_labels(all_labels, nearest_neighbors):
    recall_at_k = 0.0
    all_labels = all_labels.cpu().numpy()
    positives = 0
    for i, current_label in enumerate(all_labels):
        # print('current_label ', current_label)
        indices_of_nearest_neighbors = np.asarray(nearest_neighbors[i], dtype=int)
        labels_of_nearest_neighbors = all_labels[indices_of_nearest_neighbors]
        total_number_of_positive_pairs_for_current_label_in_the_dataset = 0
        number_of_positive_pairs_for_current_label_among_k_nearest_neighbors = 0
        for label in current_label:
            # print('label ', label)
            if label != 0:  # if some of labes == 0 than we have <3 characters on the image
                total_number_of_positive_pairs_for_current_label_in_the_dataset = \
                    total_number_of_positive_pairs_for_current_label_in_the_dataset + np.count_nonzero(
                        all_labels == label)
                number_of_positive_pairs_for_current_label_among_k_nearest_neighbors = \
                    number_of_positive_pairs_for_current_label_among_k_nearest_neighbors + np.count_nonzero(
                        labels_of_nearest_neighbors == label)
        recall_at_k = recall_at_k + \
                      number_of_positive_pairs_for_current_label_among_k_nearest_neighbors / total_number_of_positive_pairs_for_current_label_in_the_dataset
        # print('total_number_of_positive_pairs_for_current_label_in_the_dataset ', total_number_of_positive_pairs_for_current_label_in_the_dataset)
        positives = positives + total_number_of_positive_pairs_for_current_label_in_the_dataset
    # average recall at k
    recall_at_k = recall_at_k / (float(all_labels.shape[0]))
    print('total number of positive pairs ', positives)
    print('average number of relevant examples = ', positives / all_labels.shape[0])

    return recall_at_k


def get_MAP_at_k(signs_matrix, nearest_neighbors):
    MAP_at_k = 0.0
    k = nearest_neighbors.shape[1]
    signs_matrix = signs_matrix.cpu().numpy()
    for i in range(signs_matrix.shape[0]):
        indices_of_nearest_neighbors = np.asarray(nearest_neighbors[i], dtype=int)
        array_of_nearest_neighbors_signs = signs_matrix[i, indices_of_nearest_neighbors]
        # print('array_of_nearest_neighbors_signs ', array_of_nearest_neighbors_signs)
        average_precision = 0.0
        for threshold in range(1, k + 1):
            #  http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
            # if array_of_nearest_neighbors_signs[threshold - 1] == 1: # only for relevant items
            #    average_precision = average_precision + np.sum(array_of_nearest_neighbors_signs[:threshold])/threshold

            # https://habr.com/company/econtenta/blog/303458/
            average_precision = average_precision + np.sum(array_of_nearest_neighbors_signs[:threshold]) / threshold

        #  http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
        # if average_precision > 0.0:
        #    # divide only on the number of relevant examples
        #    MAP_at_k = MAP_at_k + average_precision / np.sum(array_of_nearest_neighbors_signs[:k])
        MAP_at_k = MAP_at_k + average_precision / k  # from my point of view k is more logical

    MAP_at_k = MAP_at_k / signs_matrix.shape[0]
    return MAP_at_k




def get_nearest_neighbors_from_distances_matrix(distances_matrix, k, distance_type='euclidean'):
    n = distances_matrix.shape[0]
    neighbors_lists = np.zeros((n, k))
    for i in range(n):
        if distance_type == 'cosine':
            # distances_matrix[i, i] = -np.inf
            neighbors_for_i = np.argsort(distances_matrix[i].cpu().numpy())[n - k:]  # this is for similarity
            neighbors_for_i = neighbors_for_i[::-1]  # to place the biggest scores in the beginning
        else:
            # distances_matrix[i, i] = np.inf
            neighbors_for_i = np.argsort(distances_matrix[i].cpu().numpy())[:k]  # this is for distances

        neighbors_lists[i] = neighbors_for_i
    neighbors_lists = np.asarray(neighbors_lists, dtype=int)
    print('neighbors_lists ', neighbors_lists)
    return neighbors_lists
