import cifar
import learning
import test
import net
import params
import torchvision
import utils
import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import loss
import natasha_resnet


def main():
    train_loader, test_loader = cifar.download_CIFAR100()

    ##################################################################
    #
    # Images show for debug
    #
    ##################################################################
    # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # show images
    # print("images.shape ", images.shape)
    # utils.imshow(torchvision.utils.make_grid(images)) #images = Tensor of shape (B x C x H x W)
    # print labels
    # print(' '.join('%5s' % labels[j] for j in range(params.batch_size)))

    network = net.Net(params.num_classes).cuda()

    network = natasha_resnet.resnet18(pretrained=True).cuda()

    network.fc = nn.Linear(network.fc.in_features, params.num_classes).cuda()

    ##################################################################
    #
    # Pre-training (actually fine tuning) for classification
    #
    ##################################################################

    learning.learning_process(train_loader=train_loader,
                              network=network,
                              criterion=nn.CrossEntropyLoss(),
                              test_loader=test_loader,
                              mode=params.mode_classification)

    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    network = utils.load_network_from_checkpoint(network=network,
                                                 epoch=params.number_of_epochs - 1)

    ##################################################################
    #
    # Training for representations with our margin loss
    #
    ##################################################################

    learning.learning_process(train_loader=train_loader,
                              network=network,
                              criterion=loss.margin_loss(),
                              test_loader=test_loader,
                              mode=params.mode_representation)

    test.test(test_loader=train_loader,
              network=network)
    test.test(test_loader=test_loader,
              network=network)


if __name__ == '__main__':
    main()
