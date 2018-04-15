import gc

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
from tqdm import tqdm

from utils import metric_learning_utils, params, utils


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


def fraction_of_correct_labels_in_array(actual_label, array, labels):
    correct_labels = len(np.where(labels.cpu().numpy()[array] == actual_label)[0])
    return float(correct_labels) / float(array.shape[0])


def get_total_fraction_of_correct_labels_and_total_number_of_batches(labels, neighbors_lists, number_of_outputs,
                                                                     total_fraction_of_correct_labels,
                                                                     total_number_of_batches):
    total_fraction_of_correct_labels_in_the_batch = 0
    print('number_of_outputs ', number_of_outputs)
    for i in range(number_of_outputs):
        actual_label = labels[i]
        #print('i = ', i, ' actual_label ', actual_label, 'neighbors_lists[i] ', neighbors_lists[i],
        #      ' labels.cpu().numpy()[neighbors_lists[i]] ', labels.cpu().numpy()[neighbors_lists[i]])
        fraction_of_correct_labels_among_the_k_nearest_neighbors = \
            fraction_of_correct_labels_in_array(actual_label, neighbors_lists[i], labels)

        total_fraction_of_correct_labels_in_the_batch = total_fraction_of_correct_labels_in_the_batch + \
                                                        fraction_of_correct_labels_among_the_k_nearest_neighbors
    total_number_of_batches = total_number_of_batches + 1
    fraction_for_this_batch = total_fraction_of_correct_labels_in_the_batch / float(number_of_outputs)
    total_fraction_of_correct_labels = total_fraction_of_correct_labels + fraction_for_this_batch

    return total_fraction_of_correct_labels, total_number_of_batches


def get_neighbors_lists_from_distances_matrix(distances_matrix, k, distance_type='euclidean'):
    n = distances_matrix.shape[0]
    neighbors_lists = []
    scores_lists = []
    for i in range(n):
        scores_for_i = []
        if distance_type == 'cosine':
            #distances_matrix[i, i] = -np.inf
            neighbors_for_i = np.argsort(distances_matrix[i].cpu().numpy())[n - k:]  # this is for similarity
            for j in neighbors_for_i:
                scores_for_i.append(distances_matrix[i, j])
        else:
            #distances_matrix[i, i] = np.inf
            neighbors_for_i = np.argsort(distances_matrix[i].cpu().numpy())[:k]  # this is for distances

        neighbors_lists.append(neighbors_for_i)
        scores_lists.append(scores_for_i)
    print('distance_type = ', distance_type, ' len neighbors lists ', len(neighbors_lists),
          ' neighbors_lists not from sklearn', neighbors_lists)
    print('scores_lists ', scores_lists)
    input()
    return neighbors_lists


def get_neighbors_lists(k, labels, number_of_outputs, outputs, similarity_network):
    if similarity_network is None:
        outputs = outputs.data.cpu().numpy()
        gc.collect()
        ##########################
        # For representation test we simply compute the distances while nearest neighbors search
        ##########################
        neigh = NearestNeighbors(n_neighbors=k, p=2)
        neigh.fit(outputs)
        # these are the lists of indices (inside the current batch) of the k nearest neighbors,
        # not the neighbors vectors themselves
        neighbors_lists = neigh.kneighbors(outputs, return_distance=False)
    else:
        ##########################
        # If we have learned visual similarity distances we should find nearest neighbors in another way
        #########################
        ground_truth_distances = metric_learning_utils.get_distance_matrix(outputs,
                                                                           outputs,
                                                                           distance_type=params.distance_type)
        print('ground_truth_distances ', ground_truth_distances)
        print('mean for grounf truth ', torch.mean(ground_truth_distances))
        distances_matrix = similarity_network(torch.cat((outputs,
                                                         outputs), dim=0)).data.view(params.batch_size_for_similarity,
                                                                                     params.batch_size_for_similarity)

        print('distances_matrix ', distances_matrix)
        neighbors_lists = get_neighbors_lists_from_distances_matrix(distances_matrix, k,
                                                                    distance_type=params.distance_type)

        gc.collect()

    print('neighbors_lists = ', neighbors_lists)
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


def partial_test_for_representation(k, all_outputs, all_labels, similarity_network=None):
    total_fraction_of_correct_labels = 0
    total_number_of_batches = 0

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
    #print('3617', np.sort(distances_matrix.numpy()[3617]))
    #print('3618', np.sort(distances_matrix.numpy()[3618]))
    #print('3619', np.sort(distances_matrix.numpy()[3619]))
    #print('3620', np.sort(distances_matrix.numpy()[3620]))
    #print('3621', np.sort(distances_matrix.numpy()[3621]))
    #print('0', np.sort(distances_matrix.numpy()[0]))
    #print('1', np.sort(distances_matrix.numpy()[0]))
    neighbors_lists = get_neighbors_lists_from_distances_matrix(distances_matrix, k, distance_type=params.distance_type)

    # here we add new values for current batch to the given
    # total_fraction_of_correct_labels and total_number_of_batches
    print('all_labels before fraction ', all_labels.numpy())
    total_fraction_of_correct_labels, total_number_of_batches = \
        get_total_fraction_of_correct_labels_and_total_number_of_batches(all_labels,
                                                                         neighbors_lists,
                                                                         distances_matrix.numpy().shape[0],
                                                                         total_fraction_of_correct_labels,
                                                                         total_number_of_batches)

    recall_at_k = float(total_fraction_of_correct_labels) / float(total_number_of_batches)
    print('recall_at_', k, ' of the network on the ', total_number_of_batches, ' batches: %f ' % recall_at_k)

    return recall_at_k


from sklearn.metrics import f1_score


def test_for_binary_classification(test_loader, network):
    f1 = 0.0
    number_of_small_matrices = 0.0
    for data_1 in test_loader:
        for data_2 in test_loader:
            inputs_1, labels_1 = data_1
            inputs_2, labels_2 = data_2
            # we need pairs of images in our batch
            input_pairs = Variable(torch.cat((inputs_1.view(params.batch_size_for_binary_classification, -1),
                                              inputs_2.view(params.batch_size_for_binary_classification, -1)))).cuda()
            # and +1/-1 labels matrix

            labels_matrix = Variable(utils.get_labels_matrix(labels_1, labels_2)).cuda()

            # here we should create input pair for the network from just inputs
            # print('input_pairs', input_pairs)
            outputs = network(input_pairs).data.cpu().numpy()

            ground_truth = labels_matrix.long().view(-1, 1).squeeze().data.cpu().numpy()

            #print('outputs ', outputs)
            #print('ground_truth ', ground_truth)
            outputs[np.where(outputs[:, 0] >= 0.5)[0], 0] = 1
            outputs[np.where(outputs[:, 0] < 0.5)[0], 0] = 0
            #if 1 in outputs[:, 0]:
            #    print('outputs ', outputs)
            f1 = f1 + f1_score(ground_truth, outputs[:, 0])
            number_of_small_matrices = number_of_small_matrices + 1.0
        average_f1 =  f1/number_of_small_matrices
        print('average f1 = ', average_f1)
        return average_f1


def test_for_binary_classification_1_batch(test_loader, network):
    f1 = 0.0
    number_of_small_matrices = 0.0
    ground_truth_sum = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        # and +1/-1 labels matrix
        labels_matrix = Variable(utils.get_labels_matrix(labels, labels)).cuda()

        # here we should create input pair for the network from just inputs
        # print('input_pairs', input_pairs)
        outputs = network(Variable(inputs).cuda()).data
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        # print('predicted ', predicted)

        ground_truth = labels_matrix.long().view(-1, 1).squeeze().data.cpu().numpy()
        ground_truth_sum = ground_truth_sum + np.sum(ground_truth)
        total = total + ground_truth.shape[0]
        f1 = f1 + f1_score(ground_truth, predicted)
        number_of_small_matrices = number_of_small_matrices + 1.0

    average_f1 =  f1/number_of_small_matrices
    print('average f1 = ', average_f1)
    print('ground_truth_sum/total =', ground_truth_sum/total)
    return average_f1


