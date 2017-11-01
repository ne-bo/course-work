from torch.utils.data.sampler import Sampler
import torch
import numpy as np
from torch.utils.data import TensorDataset
import cifar
import torchvision.datasets as datasets
import params
import torch.utils.data as data
import torchvision.transforms as transforms


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

    print('start_batches_indices = ', start_batches_indices)
    np.random.shuffle(start_batches_indices)
    print('after shuffling start_batches_indices = ', start_batches_indices)
    new_array = np.zeros(n, dtype=int)
    j = 0
    for i in start_batches_indices:
        new_array[j: j + batch_size] = array[i: i + batch_size]
        j = j + batch_size
    return new_array


class SmartSampler(Sampler):
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
                                   sampler=SmartSampler(train, batch_size=params.batch_size, percentage=50),
                                   num_workers=2)
    for i, d in enumerate(train_loader, 0):
        # get the inputs
        # inputs are [torch.FloatTensor of size 4x3x32x32]
        # labels are [torch.LongTensor of size 4]
        # here 4 is a batch size and 3 is a number of channels in the input images
        # 32x32 is a size of input image
        inputs, labels = d

# test_sample()
