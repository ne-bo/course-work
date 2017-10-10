import torchvision.transforms as transforms
import params
import torch.utils.data as data
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
from sklearn import model_selection


# I use this class to create a dataset which consist of CIFAR100 splited on train and test such that
# - train contains images of the first 50 classes
# - test contains images of the second 50 classes
# the code of this class is mostly copy-pasted from CIFAR10 class
# in order to be consistent with the tutorial
class CIFAR_50_50(data.Dataset):
    """CIFAR 50 50 dataset."""

    def __init__(self, train, test, train_set=True, transform=None):
        """
        Args:
            train: train for ordinary CIFAR100
            test: train for ordinary CIFAR100
        """
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

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train_set:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train_set:
            return len(self.train_data)
        else:
            return len(self.test_data)


class CIFAR_50(data.Dataset):
    def __init__(self, cifar_50_50, train_set=True, transform=None, test_size=0.2):
        """
        Args:
           cifar_50_50: train for ordinary CIFAR_50_50
        """
        self.train_set = train_set  # training set or test set
        self.transform = transform

        self.train_data, self.test_data, self.train_labels, self.test_labels = model_selection.train_test_split(
            cifar_50_50.train_data,
            cifar_50_50.train_labels,
            random_state=42,
            stratify=cifar_50_50.train_labels,
            test_size=test_size)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train_set:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train_set:
            return len(self.train_data)
        else:
            return len(self.test_data)


def download_CIFAR100():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    # download existing train and test datasets where all classes are presented in train and in test
    train = datasets.CIFAR100(root=params.data_folder,
                              train=True,
                              download=True,
                              transform=transform)

    test = datasets.CIFAR100(root=params.data_folder,
                             train=False,
                             download=True,
                             transform=transform)

    new_train_dataset = CIFAR_50_50(train,
                                    test,
                                    train_set=True,
                                    transform=transform)
    new_test_dataset = CIFAR_50_50(train,
                                   test,
                                   train_set=False,
                                   transform=transform)

    train_loader = data.DataLoader(new_train_dataset,
                                   batch_size=params.batch_size,
                                   shuffle=True,
                                   num_workers=2)
    test_loader = data.DataLoader(new_test_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=False,
                                  num_workers=2)

    new_train_dataset_for_classification = CIFAR_50(new_train_dataset,
                                                    train_set=True,
                                                    transform=transform)
    new_test_dataset_for_classification = CIFAR_50(new_test_dataset,
                                                   train_set=False,
                                                   transform=transform)
    train_loader_for_classification = data.DataLoader(new_train_dataset_for_classification,
                                                      batch_size=params.batch_size,
                                                      shuffle=True,
                                                      num_workers=2)
    test_loader_for_classification = data.DataLoader(new_test_dataset_for_classification,
                                                     batch_size=params.batch_size,
                                                     shuffle=False,
                                                     num_workers=2)

    print('train ', train.__len__())
    print('test ', test.__len__())

    print('new_train_dataset ', new_train_dataset.__len__())
    print('new_test_dataset ', new_test_dataset.__len__())

    print('new_train_dataset_for_classification ', new_train_dataset_for_classification.__len__())
    print('new_test_dataset_for_classification', new_test_dataset_for_classification.__len__())

    return train_loader, test_loader, train_loader_for_classification, test_loader_for_classification
