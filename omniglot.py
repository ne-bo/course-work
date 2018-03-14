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
        images_labels.append(labels)
        all_possible_labels.extend(labels)

    # convert string labels to numeric
    all_possible_labels = np.asarray(list(set(all_possible_labels)))
    for labels in images_labels:
        for i, label in enumerate(labels):
            #print('np.where(all_possible_labels == str(label))[0][0] ', np.where(all_possible_labels == str(label))[0][0].dtype)
            labels[i] = (np.where(all_possible_labels == str(label))[0][0])
            #print('labels[i]', labels[i].dtype)
        labels = np.array(labels)


    for i in range(100000):
        path = data_folder + ('natasha_omniglot_%s/%d.jpg' % (test_or_train, i))
        images_paths.append(path)
        images_indices.append(i)
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
    new_train_dataset = Omniglot(data_folder=data_folder,
                                 transform=transform_train,
                                 test_or_train='train'
                                 )
    new_test_dataset = Omniglot(data_folder=data_folder,
                                transform=transform_test,
                                test_or_train='test'
                                )
    print('len(new_train_dataset.train_images) ', len(new_train_dataset.train_images))
    print('len(new_train_dataset.test_images) ', len(new_train_dataset.test_images))

    print('len(new_test_dataset.train_images) ', len(new_test_dataset.train_images))
    print('len(new_test_dataset.test_images) ', len(new_test_dataset.test_images))

    return new_test_dataset, new_train_dataset


def download_Omniglot_for_representation(data_folder):
    transform_train, transform_test = create_transformations_for_test_and_train()
    new_test_dataset, new_train_dataset = create_new_train_and_test_datasets(transform_train, transform_test,
                                                                             data_folder)

    ###########################
    #
    # Attention! Here we use special UNIFORM SAMPLER!
    #
    ###########################
    train_loader = data.DataLoader(new_train_dataset,
                                   batch_sampler=BatchSampler(
                                       sampler=UniformSampler(new_train_dataset,
                                                              batch_size=params.batch_size_for_binary_classification,
                                                              number_of_samples_with_the_same_label_in_the_batch=
                                                              params.number_of_samples_with_the_same_label_in_the_batch_for_binary,
                                                              several_labels=True),
                                       batch_size=params.batch_size_for_binary_classification,
                                       drop_last=True),
                                   num_workers=8)
    print('train_loader.batch_size = ', train_loader.batch_size,
          ' train_loader.batch_sampler.batch_size =', train_loader.batch_sampler.batch_size,

          ' train_loader.dataset ', train_loader.dataset)
    print('new_test_dataset.images_paths', new_test_dataset.images_paths)
    print('new_test_dataset.images_labels', new_test_dataset.images_labels)
    print('ful batch size = ', len(new_test_dataset.test_labels))
    test_loader = data.DataLoader(new_test_dataset,

                                  # unfortunately we don't have enough memory to evaluate easily on FULL test
                                  batch_size=params.batch_size_for_representation,

                                  drop_last=True, # we need to drop last batch because it can had length less than k
                                  # and we won't be able to calculate recall at k
                                  shuffle=True, # shuffle is extremely importatnt here because we take 10 neighbors
                                  # out of 16 images in the batch
                                  num_workers=2)

    print('new_train_dataset ', new_train_dataset.__len__())
    print('new_test_dataset ', new_test_dataset.__len__())
    print('new_train_dataset.images_paths', new_train_dataset.images_paths)
    print('new_train_dataset.images_labels', new_train_dataset.images_labels)
    print('ful batch size = ', len(new_train_dataset.test_labels))

    return train_loader, test_loader
