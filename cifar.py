import torchvision.transforms as transforms
import params
import torch.utils.data as data
import torchvision.datasets as datasets
import numpy as np
from PIL import Image


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


def download_CIFAR100():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    # try transfom from good example of cifar + resnet 18
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

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

    return train_loader, test_loader
