import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

from datasets.sampling import UniformSampler
from utils import params


def get_filenames_and_labels(data_folder, test_or_train='test'):
    images_paths = []
    images_indices = []
    images_labels = []
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []

    if test_or_train == 'train':
        for i in range(1275):
            for j in range(4):
                path = data_folder + ('/ukbench%05d.jpg' % (i * 4 + j))
                images_paths.append(path)
                images_indices.append(i)
                train_images.append(path)
                train_labels.append(i)
        images_labels = train_labels

    if test_or_train == 'test':
        for i in range(1275, 2550):
            for j in range(4):
                path = data_folder + ('/ukbench%05d.jpg' % (i * 4 + j))
                images_paths.append(path)
                images_indices.append(i + 1275)
                test_images.append(path)
                test_labels.append(i)
        images_labels = test_labels

    images_labels = np.array(images_labels)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    print('images_labels.shape ', images_labels.shape)

    images_indices = np.array(images_indices)
    train_images = np.array(train_images)
    test_images = np.array(test_images)

    return images_indices, images_labels, images_paths, train_images, train_labels, test_images, test_labels


class UKB(Dataset):
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
            # print('Grayscale image is found! ', self.images_paths[index])
            image = transform_for_correction(image)
            image = transforms.ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))
            image = self.transform(image)
            # print('new image.shape ', image.shape)

        if image.shape[1] < params.initial_image_size or image.shape[2] < params.initial_image_size:
            print('image is too small', image.shape)

        return image, label


def create_transformations_for_test_and_train():
    transform_train = transforms.Compose([
        transforms.Scale(params.initial_image_scale_size),
        transforms.RandomCrop(params.initial_image_size, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Scale(params.initial_image_scale_size),
        transforms.CenterCrop(params.initial_image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_test, transform_train


def create_new_train_and_test_datasets(transform_train, transform_test, data_folder):
    # create new dataset for representational learning
    # where in train we have first 100 classes and in test the remaining 100
    new_train_dataset = UKB(data_folder=data_folder,
                                 transform=transform_train,
                                 test_or_train='train'
                                 )
    new_test_dataset = UKB(data_folder=data_folder,
                                transform=transform_test,
                                test_or_train='test'
                                )
    print('len(new_train_dataset.train_images) ', len(new_train_dataset.train_images))
    print('len(new_train_dataset.test_images) ', len(new_train_dataset.test_images))

    print('len(new_test_dataset.train_images) ', len(new_test_dataset.train_images))
    print('len(new_test_dataset.test_images) ', len(new_test_dataset.test_images))

    return new_test_dataset, new_train_dataset


def download_UKB_for_representation(data_folder):
    transform_train, transform_test = create_transformations_for_test_and_train()
    new_test_dataset, new_train_dataset = create_new_train_and_test_datasets(transform_train, transform_test,
                                                                             data_folder)

    train_loader = data.DataLoader(new_train_dataset,
                                   batch_sampler=BatchSampler(
                                       sampler=UniformSampler(new_train_dataset,
                                                              batch_size=params.batch_size_for_representation,
                                                              number_of_samples_with_the_same_label_in_the_batch=
                                                              params.number_of_samples_with_the_same_label_in_the_batch),
                                       batch_size=params.batch_size_for_representation,
                                       drop_last=True),
                                   num_workers=2)
    print('train_loader.batch_size = ', train_loader.batch_size,
          ' train_loader.batch_sampler.batch_size =', train_loader.batch_sampler.batch_size,

          ' train_loader.dataset ', train_loader.dataset)
    print('new_test_dataset.images_paths', new_test_dataset.images_paths)
    print('new_test_dataset.images_labels', new_test_dataset.images_labels)
    print('ful batch size = ', len(new_test_dataset.test_labels))
    test_loader = data.DataLoader(new_test_dataset,

                                  # unfortunately we don't have enough memory to evaluate easily on FULL test
                                  batch_size=params.batch_size_for_representation,

                                  drop_last=True,  # we need to drop last batch because it can had length less than k
                                  # and we won't be able to calculate recall at k
                                  shuffle=True,  # shuffle is extremely importatnt here because we take 10 neighbors
                                  # out of 16 images in the batch
                                  num_workers=2)

    print('new_train_dataset ', new_train_dataset.__len__())
    print('new_test_dataset ', new_test_dataset.__len__())
    print('new_train_dataset.images_paths', new_train_dataset.images_paths)
    print('new_train_dataset.images_labels', new_train_dataset.images_labels)
    print('ful batch size = ', len(new_train_dataset.test_labels))

    return train_loader, test_loader


def download_UKB_for_evaluation_or_spoc(data_folder):
    transform_train, transform_test = create_transformations_for_test_and_train()
    new_test_dataset, new_train_dataset = create_new_train_and_test_datasets(transform_train, transform_test,
                                                                             data_folder)

    # loaders with NO shuffling!!!!
    train_loader = data.DataLoader(new_train_dataset,
                                   batch_size=params.batch_size_for_representation,
                                   num_workers=2)
    print('loaders with NO shuffling train_loader.batch_size = ', train_loader.batch_size,
          ' loaders with NO shuffling train_loader.batch_sampler.batch_size =', train_loader.batch_sampler.batch_size,

          ' loaders with NO shuffling train_loader.dataset ', train_loader.dataset)
    test_loader = data.DataLoader(new_test_dataset,
                                  batch_size=params.batch_size_for_representation,
                                  num_workers=2)

    return train_loader, test_loader
