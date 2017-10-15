from torch.autograd import Variable
import torch
from sklearn.neighbors import NearestNeighbors
import datetime


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
    correct_labels = 0
    k = array.shape[0]
    for i in range(k):
        if labels[array[i]] == actual_label:
            correct_labels = correct_labels + 1
    return float(correct_labels) / float(k)


def test_for_representation(test_loader, network, k):
    total_fraction_of_correct_labels = 0
    total_number_of_outputs = 0

    for data in test_loader:
        images, labels = data
        outputs = network(Variable(images).cuda())

        # print('outputs = ', outputs)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(outputs.data.cpu().numpy())
        neighbors_list = neigh.kneighbors(outputs.data.cpu().numpy(), return_distance=False)
        number_of_outputs = outputs.data.shape[0]
        for i in range(number_of_outputs):
            actual_label = labels[i]

            fraction_of_correct_labels_among_the_k_nearest_neighbors = \
                fraction_of_correct_labels_in_array(actual_label, neighbors_list[i], labels)
            total_fraction_of_correct_labels = total_fraction_of_correct_labels + \
                                               fraction_of_correct_labels_among_the_k_nearest_neighbors
            total_number_of_outputs = total_number_of_outputs + number_of_outputs

    recall_at_k = float(total_fraction_of_correct_labels) / float(total_number_of_outputs)
    print('recall_at_k of the network on the ', total_number_of_outputs, ' images: %f ' % recall_at_k)

    return recall_at_k
