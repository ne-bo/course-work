import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

import params


# This function takes main label and return an array of indices
# In this arrray we have such structures
# [main_label, main_label, ...., main_lable, other_label, other_label, ..., other_label]
# this sctucture contains percentage of other_label and has a length of batch size
# Returned array is a
# [structure1, structure2, ...., structureN]
# So from returned array we can have a lot of batches with our main label represented
def get_indices_for_main_label(main_class, train_labels, number_of_main_class_samples,
                               number_of_other_classes_samples):
    indices_of_main_class = np.where(train_labels == main_class)
    indices_of_other_classes = np.where(train_labels != main_class)

    np.random.shuffle(indices_of_main_class[0])
    np.random.shuffle(indices_of_other_classes[0])

    shuffled_main_class = indices_of_main_class[0]
    shuffled_other_classes = indices_of_other_classes[0]

    print('shuffled_main_class.shape = ', shuffled_main_class.shape)
    print('shuffled_other_classes.shape = ', shuffled_other_classes.shape)
    new_indices_to_take = np.empty(0, dtype=int)
    for i in range(len(shuffled_main_class) // number_of_main_class_samples - 1):
        new_indices_to_take = np.hstack(
            (new_indices_to_take,
             shuffled_main_class[i * number_of_main_class_samples: (i + 1) * number_of_main_class_samples],
             shuffled_other_classes[i * number_of_other_classes_samples:(i + 1) * number_of_other_classes_samples]))
    return new_indices_to_take


def shuffle_with_batch_size(array, batch_size):
    n = array.shape[0]
    assert n % batch_size == 0

    start_batches_indices = np.arange(start=0, stop=array.shape[0], step=batch_size)

    #print('start_batches_indices = ', start_batches_indices)
    np.random.shuffle(start_batches_indices)
    #print('after shuffling start_batches_indices = ', start_batches_indices)
    new_array = np.zeros(n, dtype=int)
    j = 0
    for i in start_batches_indices:
        new_array[j: j + batch_size] = array[i: i + batch_size]
        j = j + batch_size
    return new_array


class PercentageSampler(Sampler):
    """Samples elements with majority of samples with the same label
    Arguments:
        percentage (int) : percentage of samples with the same label among all sampled elements
    """

    def __init__(self, data_source, batch_size, percentage):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.batch_size = batch_size
        self.percentage = percentage

    # here we stacks arrays of batches with different main labels
    def __iter__(self):
        train_labels = np.array(self.data_source.train_labels)
        number_of_main_class_samples = int(self.batch_size * self.percentage // 100)
        number_of_other_classes_samples = self.batch_size - number_of_main_class_samples

        all_labels = np.array(list(set(train_labels)))
        np.random.shuffle(all_labels)
        indices_to_take = np.empty(0, dtype=int)
        for i in all_labels:
            main_class = i
            print('current main_label = ', main_class)
            # generate indices for this label
            new_indices_to_take = get_indices_for_main_label(main_class,
                                                                  train_labels,
                                                                  number_of_main_class_samples,
                                                                  number_of_other_classes_samples)
            print('new_indices_to_take = ', new_indices_to_take.shape, ' ', new_indices_to_take)
            # add new indices to all
            indices_to_take = np.hstack(
                (indices_to_take,
                 new_indices_to_take))

        print('indices_to_take = ', indices_to_take.shape, ' ', indices_to_take)
        print('labels to take = ', self.data_source.train_labels[indices_to_take])
        shuffled_batches = shuffle_with_batch_size(indices_to_take, self.batch_size)
        print('shuffled_batches = ', shuffled_batches.shape, ' ', shuffled_batches)
        print('shuffled_batches labels = ', self.data_source.train_labels[shuffled_batches])
        return iter(shuffled_batches)



    def __len__(self):
        return self.batch_size


class UniformSampler(Sampler):
    """Samples elements with roughly uniform distribution of samples with the same label
    Arguments:
        percentage (int) : percentage of samples with the same label among all sampled elements
    """

    def __init__(self, data_source, batch_size, number_of_samples_with_the_same_label_in_the_batch,
                 several_labels=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.batch_size = batch_size
        self.number_of_samples_with_the_same_label_in_the_batch = number_of_samples_with_the_same_label_in_the_batch
        self.several_labels = several_labels

    def get_indices_for_new_batch(self, remaining):
        new_batch = np.empty(0, dtype=int)

        all_remaining_labels = np.array(list(set(remaining)))
        np.random.shuffle(all_remaining_labels)

        while new_batch.shape[0] < self.batch_size:
            # get new label
            label = all_remaining_labels[0]
            #print('label = ', label)
            all_remaining_labels = all_remaining_labels[1:]
            np.random.shuffle(all_remaining_labels)
            # get number of samples for current label in this batch
            number_of_sumples = np.random.randint(low=int(self.number_of_samples_with_the_same_label_in_the_batch * 0.75),
                                                  high=int(self.number_of_samples_with_the_same_label_in_the_batch * 1.25))
            #print('number_of_sumples = ', number_of_sumples)
            # we will put this indices in our new batch
            indices_to_put_in_new_batch = np.random.permutation(np.where(remaining == label)[0])[:number_of_sumples]
            new_batch = np.hstack((new_batch, indices_to_put_in_new_batch))
            # and remove them from the remaining part
            #print('puted ', remaining[indices_to_put_in_new_batch].shape)


        # and remove the tail from the batch
        new_batch = new_batch[:self.batch_size]
        #print('new_batch ', np.sort(remaining[new_batch]), ' new_batch.shape ', new_batch.shape)
        #print('remaining ', remaining.shape)

        return new_batch, remaining

    # here we stacks arrays of batches with different main labels
    def __iter__(self):
        if self.several_labels:
            # if we can have several labels for the 1 image we just take the random label
            train_labels = np.array(self.data_source.train_labels)
            for i, labels in enumerate(train_labels):
                number_of_different_labels = labels.shape[0]
                random_index = np.random.randint(low=0, high=number_of_different_labels)
                train_labels[i] = labels[random_index]
            print('several train_labels', train_labels)
        else:
            train_labels = np.array(self.data_source.train_labels)

        indices_to_take = np.empty(0, dtype=int)
        remaining = train_labels
        number_of_batches = 0
        while number_of_batches < len(train_labels)// self.batch_size: #len(remaining) > self.batch_size:
            new_batch, remaining = self.get_indices_for_new_batch(remaining)
            # add new indices to all
            indices_to_take = np.hstack(
                (indices_to_take,
                 new_batch))
            number_of_batches = number_of_batches + 1

        print('indices_to_take = ', indices_to_take.shape, ' ', indices_to_take)
        #  print('labels to take = ', self.data_source.train_labels[indices_to_take])

        shuffled_batches = shuffle_with_batch_size(indices_to_take, self.batch_size)

        return iter(shuffled_batches)



    def __len__(self):
        return self.batch_size







def test_sample():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    train = datasets.CIFAR100(root=params.data_folder,
                              train=True,
                              download=True,
                              transform=transform)
    train_loader = data.DataLoader(train,
                                   batch_size=params.batch_size,
                                   sampler=PercentageSampler(train, batch_size=params.batch_size, percentage=50),
                                   num_workers=2)
    for i, d in enumerate(train_loader, 0):
        # get the inputs
        # inputs are [torch.FloatTensor of size 4x3x32x32]
        # labels are [torch.LongTensor of size 4]
        # here 4 is a batch size and 3 is a number of channels in the input images
        # 32x32 is a size of input image
        inputs, labels = d

# test_sample()
