from torch.autograd import Variable
import torch
from sklearn.neighbors import NearestNeighbors
import datetime
import numpy as np

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


def test_for_representation(test_loader, network, k):
    total_fraction_of_correct_labels = 0
    total_number_of_batches = 0

    for data in test_loader:
        images, labels = data
        outputs = network(Variable(images).cuda())

        # print('outputs = ', outputs)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(outputs.data.cpu().numpy())
        neighbors_list = neigh.kneighbors(outputs.data.cpu().numpy(), return_distance=False)
        number_of_outputs = outputs.data.shape[0]
        total_fraction_of_correct_labels_in_the_batch = 0
        for i in range(number_of_outputs):
            actual_label = labels[i]
            #print('actual_label', actual_label)
            #print('neighbors_list[i]', neighbors_list[i])

            fraction_of_correct_labels_among_the_k_nearest_neighbors = \
                fraction_of_correct_labels_in_array(actual_label, neighbors_list[i], labels)

            #print('fraction_of_correct_labels_among_the_k_nearest_neighbors ', fraction_of_correct_labels_among_the_k_nearest_neighbors)
            total_fraction_of_correct_labels_in_the_batch = total_fraction_of_correct_labels_in_the_batch + \
                                               fraction_of_correct_labels_among_the_k_nearest_neighbors
        total_number_of_batches = total_number_of_batches + 1

        #print('total_fraction_of_correct_labels_in_the_batch = ', total_fraction_of_correct_labels_in_the_batch)
        #print('number_of_outputs ', number_of_outputs)
        fraction_for_this_batch = total_fraction_of_correct_labels_in_the_batch/float(number_of_outputs)
        total_fraction_of_correct_labels = total_fraction_of_correct_labels + fraction_for_this_batch

    recall_at_k = float(total_fraction_of_correct_labels) / float(total_number_of_batches)
    print('recall_at_', k, ' of the network on the ', total_number_of_batches, ' batches: %f ' % recall_at_k)

    return recall_at_k
