import gc

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm

from utils import params, utils


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
        labels_matrix = Variable(utils.get_labels_matrix(labels, labels)).cuda()

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
def recall_test_for_representation(k, all_outputs, all_labels, similarity_network=None):
    distances_matrix = compute_the_distance_matrix_with_network(all_outputs, similarity_network)
    nearest_neighbors = get_nearest_neighbors_from_distances_matrix(distances_matrix, k,
                                                                    distance_type=params.distance_type)

    # here we add new values for current batch to the given
    # total_fraction_of_correct_labels and total_number_of_batches
    print('all_labels before fraction ', all_labels.numpy())

    recall_at_k = get_recall(all_labels, nearest_neighbors)
    print('recall_at_', k, ' of the network: %f ' % recall_at_k)

    return recall_at_k


#################################################
#
# Helper functions
#
#################################################

def compute_the_distance_matrix_with_network(all_outputs, similarity_network):
    number_of_outputs = all_outputs.shape[0]
    number_of_batches = all_outputs.shape[0] // params.batch_size_for_similarity
    print('number_of_batches = ', number_of_batches)

    distances_matrix = torch.from_numpy(np.zeros((number_of_outputs, number_of_outputs))).float()
    for i in tqdm(range(number_of_batches)):
        for j in range(number_of_batches):
            representation_outputs_1 = all_outputs[
                                       i * params.batch_size_for_similarity:
                                       (i + 1) * params.batch_size_for_similarity]
            representation_outputs_2 = all_outputs[
                                       j * params.batch_size_for_similarity:
                                       (j + 1) * params.batch_size_for_similarity]
            similarity_outputs = similarity_network(Variable(torch.cat(
                (representation_outputs_1,
                 representation_outputs_2), dim=0))).view(params.batch_size_for_similarity,
                                                          params.batch_size_for_similarity)

            distances_matrix[
            i * params.batch_size_for_similarity:(i + 1) * params.batch_size_for_similarity,
            j * params.batch_size_for_similarity:(j + 1) * params.batch_size_for_similarity] = similarity_outputs.data
            gc.collect()
    print('full distances_matrix', distances_matrix.numpy().shape)
    return distances_matrix


def get_recall(all_labels, nearest_neighbors):
    recall_at_k = 0.0
    for i, current_label in enumerate(all_labels):
        total_number_of_this_label_in_the_dataset = np.count_nonzero(all_labels == current_label)
        indices_of_nearest_neighbors = nearest_neighbors[i]
        labels_of_nearest_neighbors = all_labels[indices_of_nearest_neighbors]
        number_of_this_label_among_k_nearest_neighbors = np.count_nonzero(labels_of_nearest_neighbors == current_label)
        recall_at_k = recall_at_k + \
                      number_of_this_label_among_k_nearest_neighbors / total_number_of_this_label_in_the_dataset

    return recall_at_k


def get_nearest_neighbors_from_distances_matrix(distances_matrix, k, distance_type='euclidean'):
    n = distances_matrix.shape[0]
    neighbors_lists = np.zeros((n, k))
    for i in range(n):
        if distance_type == 'cosine':
            # distances_matrix[i, i] = -np.inf
            neighbors_for_i = np.argsort(distances_matrix[i].cpu().numpy())[n - k:]  # this is for similarity
        else:
            # distances_matrix[i, i] = np.inf
            neighbors_for_i = np.argsort(distances_matrix[i].cpu().numpy())[:k]  # this is for distances

        neighbors_lists[i] = neighbors_for_i
    return neighbors_lists
