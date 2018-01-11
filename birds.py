import torchvision.transforms as transforms
import params
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler
import torchvision.datasets as datasets
import numpy as np
from sklearn import model_selection
from sampling import PercentageSampler, UniformSampler
from torch.utils.data import Dataset, TensorDataset
import torch
from PIL import Image


def get_filenames_and_labels(data_folder, test_or_train='test'):
    f_i = open(data_folder + '/CUB_200_2011/images.txt', "r")
    f_l = open(data_folder + '/CUB_200_2011/image_class_labels.txt', "r")
    lines_i = f_i.readlines()
    lines_l = f_l.readlines()
    images_paths = []
    images_indices = []
    images_labels = []
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []

    for x in lines_l:
        label = int(x.split(' ')[1])
        images_labels.append(label)
        if test_or_train == 'train' and label <= 100:
            train_labels.append(label)
        if test_or_train == 'test' and label > 100:
            test_labels.append(label)
    images_labels = np.array(images_labels)

    print('images_labels.shape ', images_labels.shape)
    for x in lines_i:
        path = data_folder + '/CUB_200_2011/images/' + test_or_train + '100/' + (x.split(' ')[1]).strip()
        index = int(x.split(' ')[0]) - 1

        if test_or_train == 'train' and images_labels[index] <= 100 and index <= 5863:
            images_paths.append(path)
            images_indices.append(index)
            train_images.append(path)
        if test_or_train == 'test' and images_labels[index] > 100 and index > 5836:
            images_paths.append(path)
            images_indices.append(index)
            test_images.append(path)

    f_i.close()
    f_l.close()

    images_indices = np.array(images_indices)
    train_images = np.array(train_images)
    test_images = np.array(test_images)

    return images_indices, images_labels, images_paths, train_images, train_labels, test_images, test_labels


class BIRDS100(Dataset):
    def __init__(self, data_folder, transform=None, test_or_train='test'):
        self.data_folder = data_folder
        self.transform = transform
        if test_or_train == 'train':
            self.train = True
        else:
            self.train = False
        self.images_indices, \
        self.images_labels, \
        self.images_paths, \
        self.train_images, \
        self.train_labels, \
        self.test_images, \
        self.test_labels = get_filenames_and_labels(data_folder,
                                                    test_or_train=test_or_train)

        print('self.images_labels ', self.images_labels)

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, index):
        transform_for_correction = transforms.Compose([
            transforms.ToPILImage(),
        ])
        if self.train:
            image = self.transform(Image.open(self.train_images[index]))
            label = self.train_labels[index]
        else:
            image = self.transform(Image.open(self.test_images[index]))
            label = self.test_labels[index]

        if image.shape[0] == 1:
            print('Grayscale image is found! ', self.images_paths[index])
            image = transform_for_correction(image)
            image = transforms.ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))
            image = self.transform(image)
            print('new image.shape ', image.shape)

        if image.shape[1] < 224 or image.shape[2] < 224:
            print('image is too small', image.shape)

        return image, label


def create_transformations_for_test_and_train():
    transform_train = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_test, transform_train


def create_new_train_and_test_datasets(transform_train, transform_test, data_folder):
    # create new dataset for representational learning
    # where in train we have first 100 classes and in test the remaining 100
    new_train_dataset = BIRDS100(data_folder=data_folder,
                                 transform=transform_train,
                                 test_or_train='train'
                                 )
    new_test_dataset = BIRDS100(data_folder=data_folder,
                                transform=transform_test,
                                test_or_train='test'
                                )
    print('len(new_train_dataset.train_images) ', len(new_train_dataset.train_images))
    print('len(new_train_dataset.test_images) ', len(new_train_dataset.test_images))

    print('len(new_test_dataset.train_images) ', len(new_test_dataset.train_images))
    print('len(new_test_dataset.test_images) ', len(new_test_dataset.test_images))

    return new_test_dataset, new_train_dataset


def download_BIRDS_for_classification(data_folder):
    transform_test, transform_train = create_transformations_for_test_and_train()

    test_dataset_for_classification, train_dataset_for_classification = create_new_train_and_test_datasets(
        transform_train,
        transform_test,
        data_folder)

    train_loader_for_classification = data.DataLoader(train_dataset_for_classification,
                                                      batch_size=params.batch_size,
                                                      shuffle=True,
                                                      num_workers=2)
    test_loader_for_classification = data.DataLoader(train_dataset_for_classification,  # here for preclassification
                                                     # we just take train and test sets the same
                                                     # containing first 100 classes
                                                     batch_size=params.batch_size,
                                                     shuffle=False,

                                                     num_workers=2)

    print('new_train_dataset_for_classification ', train_dataset_for_classification.__len__())
    print('new_test_dataset_for_classification', test_dataset_for_classification.__len__())

    return train_loader_for_classification, test_loader_for_classification


def download_BIRDS_for_representation(data_folder):
    transform_train, transform_test = create_transformations_for_test_and_train()
    new_test_dataset, new_train_dataset = create_new_train_and_test_datasets(transform_train, transform_test,
                                                                             data_folder)

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
    print('new_test_dataset.images_paths', new_test_dataset.images_paths)
    print('new_test_dataset.images_labels', new_test_dataset.images_labels)
    test_loader = data.DataLoader(new_test_dataset,
                                  batch_size=params.batch_size,
                                  drop_last=True, # we need to drop last batch because it can had length less than k
                                  # and we won't be able to calculate recall at k
                                  shuffle=True, # shuffle is extremely importatnt here because we take 10 neighbors
                                  # out of 16 images in the batch
                                  num_workers=2)

    print('new_train_dataset ', new_train_dataset.__len__())
    print('new_test_dataset ', new_test_dataset.__len__())

    return train_loader, test_loader
