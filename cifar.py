import torchvision.transforms as transforms
import params
import torch.utils.data as data
import torchvision.datasets as datasets


def download_CIFAR100():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    train = datasets.CIFAR100(root=params.data_folder,
                              train=True,
                              download=True,
                              transform=transform)
    train_loader = data.DataLoader(train,
                                   batch_size=params.batch_size,
                                   shuffle=True,
                                   num_workers=2)

    test = datasets.CIFAR100(root=params.data_folder,
                             train=False,
                             download=True,
                             transform=transform)
    test_loader = data.DataLoader(test,
                                  batch_size=params.batch_size,
                                  shuffle=False,
                                  num_workers=2)

    return train_loader, test_loader
