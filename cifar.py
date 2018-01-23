import torchvision.transforms as transforms
import params
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler
import torchvision.datasets as datasets
import numpy as np
from sklearn import model_selection
from sampling import UniformSampler


# I use this class to create a dataset which consist of CIFAR100 splited on train and test such that
# - train contains images of the first 50 classes
# - test contains images of the second 50 classes
# the code of this class is mostly copy-pasted from CIFAR10 class
# in order to be consistent with the tutorial
class CIFAR_50_50(datasets.CIFAR100):
    """CIFAR 50 50 dataset."""

    def __init__(self, train, test, train_set=True, transform=None):
        """
        Args:
            train: train for ordinary CIFAR100
            test: train for ordinary CIFAR100
        """
        super(CIFAR_50_50, self).__init__(root=params.data_folder)

        self.train_set = train_set  # training set or test set
        self.transform = transform

        # merge test and train of CIFAR100
        new_data = np.vstack((train.train_data, test.test_data))
        new_labels = np.hstack((train.train_labels, test.test_labels))

        # create new train consisting only of images of the first 50 classes
        new_train = new_data[np.where(new_labels < 50)]
        new_train_labels = new_labels[new_labels < 50]

        # create new test consisting only of images of the last 50 classes
        new_test = new_data[np.where(new_labels >= 50)]
        new_test_labels = new_labels[new_labels >= 50]

        self.train_data = new_train
        self.test_data = new_test
        self.train_labels = new_train_labels
        self.test_labels = new_test_labels


class CIFAR_50(datasets.CIFAR100):
    def __init__(self, cifar_50_50, train_set=True, transform=None, test_size=0.2):
        """
        Args:
           cifar_50_50: train for ordinary CIFAR_50_50
        """
        super(CIFAR_50, self).__init__(root=params.data_folder)

        self.train_set = train_set  # training set or test set
        self.transform = transform

        self.train_data, self.test_data, self.train_labels, self.test_labels = model_selection.train_test_split(
            cifar_50_50.train_data,
            cifar_50_50.train_labels,
            random_state=42,
            stratify=cifar_50_50.train_labels,
            test_size=test_size)


def create_transformation_for_new_dataset():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    return transform


def create_transformations_for_test_and_train():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_test, transform_train


def download_CIFAR100_for_classification():
    transform = create_transformation_for_new_dataset()
    new_test_dataset, new_train_dataset, test, train = create_new_train_and_test_datasets(transform)

    # create a dataset for pretraining on classification task
    # where in all dataset we have first 50 classes
    transform_test, transform_train = create_transformations_for_test_and_train()

    new_train_dataset_for_classification = CIFAR_50(new_train_dataset,
                                                    train_set=True,
                                                    transform=transform_train)
    new_test_dataset_for_classification = CIFAR_50(new_test_dataset,
                                                   train_set=False,
                                                   transform=transform_test)
    train_loader_for_classification = data.DataLoader(new_train_dataset_for_classification,
                                                      batch_size=params.batch_size,
                                                      shuffle=True,
                                                      num_workers=2)
    test_loader_for_classification = data.DataLoader(new_test_dataset_for_classification,
                                                     batch_size=params.batch_size,
                                                     shuffle=False,
                                                     num_workers=2)

    print('new_train_dataset_for_classification ', new_train_dataset_for_classification.__len__())
    print('new_test_dataset_for_classification', new_test_dataset_for_classification.__len__())

    return train_loader_for_classification, test_loader_for_classification


def download_CIFAR100_for_representation():
    transform = create_transformation_for_new_dataset()
    new_test_dataset, new_train_dataset, test, train = create_new_train_and_test_datasets(transform)

    train_loader = data.DataLoader(new_train_dataset,
                                   batch_sampler=BatchSampler(
                                       sampler=UniformSampler(new_train_dataset,
                                                              batch_size=params.batch_size,
                                                              number_of_samples_with_the_same_label_in_the_batch=
                                                              params.number_of_samples_with_the_same_label_in_the_batch),
                                       batch_size=params.batch_size,
                                       drop_last=True),
                                   num_workers=2)
    print('train_loader.batch_size = ', train_loader.batch_size,
          ' train_loader.batch_sampler.batch_size =', train_loader.batch_sampler.batch_size,

          ' train_loader.dataset ', train_loader.dataset)
    test_loader = data.DataLoader(new_test_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=False,
                                  num_workers=2)

    print('train ', train.__len__())
    print('test ', test.__len__())

    print('new_train_dataset ', new_train_dataset.__len__())
    print('new_test_dataset ', new_test_dataset.__len__())

    return train_loader, test_loader


def create_new_train_and_test_datasets(transform):
    # download existing train and test datasets where all classes are presented in train and in test
    train = datasets.CIFAR100(root=params.data_folder,
                              train=True,
                              download=True,
                              transform=transform)
    test = datasets.CIFAR100(root=params.data_folder,
                             train=False,
                             download=True,
                             transform=transform)
    # create new dataset for representational learning
    # where in train we have first 50 classes and in test the remaining 50
    new_train_dataset = CIFAR_50_50(train,
                                    test,
                                    train_set=True,
                                    transform=transform)
    new_test_dataset = CIFAR_50_50(train,
                                   test,
                                   train_set=False,
                                   transform=transform)
    return new_test_dataset, new_train_dataset, test, train


def download_CIFAR10(batch_size=params.batch_size_for_cifar):
    transform_test, transform_train = create_transformations_for_test_and_train()

    # download existing train and test datasets where all classes are presented in train and in test
    train = datasets.CIFAR10(root=params.data_folder,
                             train=True,
                             download=True,
                             transform=transform_train)

    test = datasets.CIFAR10(root=params.data_folder,
                            train=False,
                            download=True,
                            transform=transform_test)

    train_loader = data.DataLoader(train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=2)
    test_loader = data.DataLoader(test,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2)

    print('train ', train.__len__())
    print('test ', test.__len__())

    return train_loader, test_loader
