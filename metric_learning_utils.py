import numpy as np
import torch
from torch.autograd import Variable

import loss


# returns a vector of pairwise cosine similarities between
# the image number i and all images from
# i + 1 till n
def get_cosine_similarities_i(i, input, n, representation_vector_length):
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    return cosine_similarity(Variable(torch.ones([n - i - 1, representation_vector_length]).cuda()) *
                             input[i].cuda(),
                             input[i + 1:].cuda())


def create_a_batch_of_pairs(representations, labels):
    n = representations.shape[0]
    representation_vector_length = representations[0].shape[0]

    labels = Variable(labels.cuda())
    representations = Variable(representations.cuda())

    batch_of_pairs = torch.cat((representations[n - 2].cuda(), representations[n - 1].cuda()), dim=0).view(1, -1)
    distances_for_pairs = loss.MarginLoss.get_distances_i(n - 2, representations, n, representation_vector_length)
    cosine_similarities = get_cosine_similarities_i(n - 2, representations, n, representation_vector_length)

    signs_for_pairs = loss.MarginLoss.get_signs_i(n - 2, labels, n)
    for i in range(n - 2):
        first_elements_of_pairs = Variable(torch.ones([n - i - 1, representation_vector_length]).cuda()) * \
                                  representations[i].cuda()
        second_elements_of_pairs = representations[i + 1:].cuda()

        pairs_with_i = torch.cat((first_elements_of_pairs, second_elements_of_pairs), dim=1)
        # print('i for batch of pairs = ', i)
        batch_of_pairs = torch.cat((batch_of_pairs, pairs_with_i), dim=0)

        distances_for_pairs = torch.cat((distances_for_pairs,
                                         loss.MarginLoss.get_distances_i(i, representations, n,
                                                                         representation_vector_length)), dim=0)
        cosine_similarities = torch.cat((cosine_similarities, get_cosine_similarities_i(i, representations, n,
                                                                         representation_vector_length)), dim=0)
        signs_for_pairs = torch.cat((signs_for_pairs, loss.MarginLoss.get_signs_i(i, labels, n)), dim=0)

    # We initialized ours batch_of_pairs, distances_for_pairs and signs_for_pairs with the pair (n - 2, n - 1)
    # but it is more beautiful and logical to have this pair in the end so let's move it in the end
    total_number_of_pairs = batch_of_pairs.data.shape[0]
    indices = [total_number_of_pairs - 1]
    indices.extend(np.arange(0, total_number_of_pairs - 1, step=1))
    indices = torch.cuda.LongTensor(np.array(indices).tolist())  # without this mystical casting to array and then to
    # list it doesn't work

    batch_of_pairs = Variable(torch.index_select(batch_of_pairs.data, dim=0, index=indices))
    distances_for_pairs = Variable(torch.index_select(distances_for_pairs.data, dim=0, index=indices))
    cosine_similarities = Variable(torch.index_select(cosine_similarities.data, dim=0, index=indices))
    signs_for_pairs = Variable(torch.index_select(signs_for_pairs.data, dim=0, index=indices))

    # Finally we have the following pairs order
    # (0, 1),         (0, 2), ............, (0, n - 2), (0, n - 1),
    # (1, 2),         (1, 3), ............, (1, n - 1),
    # ...
    # (n - 3, n - 2), (n - 3, n - 1),
    # (n - 2, n - 1)

    # reshaping of distances in signs to have 1 dimension
    distances_for_pairs = distances_for_pairs.view(distances_for_pairs.data.shape[0])
    signs_for_pairs = signs_for_pairs.view(signs_for_pairs.data.shape[0])

    return batch_of_pairs, distances_for_pairs, signs_for_pairs, cosine_similarities


def create_a_batch_of_pairs_i(representations, i):
    n = representations.data.shape[0]
    representation_vector_length = representations[0].data.shape[0]

    first_elements_of_pairs = Variable(torch.ones([n, representation_vector_length]).cuda()) * representations[i].cuda()
    second_elements_of_pairs = representations.cuda()

    pairs_with_i = torch.cat((first_elements_of_pairs, second_elements_of_pairs), dim=1)
    # print('i for batch of pairs = ', i)

    return pairs_with_i


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
