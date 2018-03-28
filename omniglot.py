import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import params
from sampling import UniformSampler


def get_filenames_and_labels(data_folder, test_or_train='test'):
    images_paths = []
    images_indices = []
    images_labels = []
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []

    f_l = open(data_folder + 'labels_natasha_omniglot_%s' % test_or_train, "r", encoding="utf-8")
    lines_l = f_l.readlines()

    all_possible_labels = []
    for x in lines_l:
        x = x.replace('\n', '')
        labels = x.split(' ')[1:]
        while len(labels) < 3:
            labels.append('1000000')
            #print('labels appended ', labels)
        images_labels.append(labels)
        all_possible_labels.extend(labels)

    # convert string labels to numeric
    # here if an image has less then 3 labels the label 0 will be addaed up to 3 labels
    # for example if the image has labels 34 and 22 then the final set of labels will be 34, 22, 0
    # if the image has only 1 character and label 2 then the final set of labels will be  2,  0, 0
    all_possible_labels = np.sort(np.asarray(list(set(all_possible_labels))))
    # print('all_possible_labels', all_possible_labels)
    for labels in images_labels:
        for i, label in enumerate(labels):
            # print('np.where(all_possible_labels == str(label))[0][0] ', np.where(all_possible_labels == str(label))[0][0].dtype)
            labels[i] = (np.where(all_possible_labels == str(label))[0][0])
            # print('labels[i]', labels[i].dtype)
        labels = np.array(labels)

    for i in range(10000):
        path = data_folder + ('natasha_omniglot_%s/%d.jpg' % (test_or_train, i))
        images_paths.append(path)
        images_indices.append(i)
        images_labels[i] = np.array(images_labels[i])
        if test_or_train == 'train':
            train_images.append(path)
            train_labels.append(images_labels[i])
        else:
            test_labels.append(images_labels[i])
            test_images.append(path)

    if test_or_train == 'train':
        images_labels = train_labels
    else:
        images_labels = test_labels

    images_labels = np.array(images_labels)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    print('images_labels.shape ', images_labels.shape)

    images_indices = np.array(images_indices)
    train_images = np.array(train_images)
    test_images = np.array(test_images)

    return images_indices, images_labels, images_paths, train_images, train_labels, test_images, test_labels


class Omniglot(Dataset):
    def __init__(self, data_folder, transform=None, test_or_train='test', image_size = 32):
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
        self.image_size = image_size

        print('self.images_labels ', self.images_labels)

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, index):
        if self.train:
            image = self.transform(Image.open(self.train_images[index]))
            # label = self.train_labels[index]
            label = self.images_labels[index]
        else:
            image = self.transform(Image.open(self.test_images[index]))
            # label = self.test_labels[index]
            label = self.images_labels[index]
        # print('image, label ', image, label)
        return image, label


def create_transformations_for_test_and_train(image_size):
    transform_train = transforms.Compose([
        transforms.Scale(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
    ])
    return transform_test, transform_train


def create_new_train_and_test_datasets(transform_train, transform_test, data_folder, image_size):
    # create new dataset for representational learning
    # where in train we have first 100 classes and in test the remaining 100
    new_train_dataset = Omniglot(data_folder=data_folder,
                                 transform=transform_train,
                                 test_or_train='train',
                                 image_size = 32
                                 )
    new_test_dataset = Omniglot(data_folder=data_folder,
                                transform=transform_test,
                                test_or_train='test',
                                image_size = 32
                                )
    print('len(new_train_dataset.train_images) ', len(new_train_dataset.train_images))
    print('len(new_train_dataset.test_images) ', len(new_train_dataset.test_images))

    print('len(new_test_dataset.train_images) ', len(new_test_dataset.train_images))
    print('len(new_test_dataset.test_images) ', len(new_test_dataset.test_images))

    return new_test_dataset, new_train_dataset


def download_Omniglot_for_representation(data_folder, image_size):
    transform_train, transform_test = create_transformations_for_test_and_train(image_size)
    new_test_dataset, new_train_dataset = create_new_train_and_test_datasets(transform_train, transform_test,
                                                                             data_folder, image_size)

    ###########################
    #
    # Attention! Here we use special UNIFORM SAMPLER!
    #
    ###########################

    uniform_sampler = UniformSampler(
        new_train_dataset,
        batch_size=params.batch_size_for_binary_classification,
        number_of_samples_with_the_same_label_in_the_batch=params.number_of_samples_with_the_same_label_in_the_batch_for_binary,
        several_labels=True
    )
    print('uniform_sampler ', uniform_sampler)
    batch_sampler = BatchSampler(
            sampler=uniform_sampler,
            batch_size=params.batch_size_for_binary_classification,
            drop_last=True
        )
    print('batch_sampler ', batch_sampler)

    print('************************************************')
    print('    Create train loared')
    print('************************************************')
    train_loader = data.DataLoader(
        new_train_dataset,
        batch_sampler=batch_sampler,
        num_workers=8
    )

    print('train_loader ', train_loader.__iter__().collate_fn)
    print('************************************************')
    print('    Create test loared')
    print('************************************************')

    test_loader = data.DataLoader(
        new_test_dataset,
        batch_size=params.batch_size_for_binary_classification,
        drop_last=True,  # we need to drop last batch because it can had length less than k
        # and we won't be able to calculate recall at k
        shuffle=True,
        num_workers=2
    )

    return train_loader, test_loader
