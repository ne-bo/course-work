import matplotlib.pyplot as plt
import numpy as np
import torch


# functions to show an image
def imshow(img):
    img = img / 2.0 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() #you need this only if you want to actually see the images


def save_checkpoint(network, optimizer, epoch, filename='checkpoint.pth.tar'):
    torch.save({
                    'epoch': epoch + 1,
                    'state_dict': network.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, filename)


def load_network_and_optimizer_from_checkpoint(network, optimizer, epoch, name_prefix_for_saved_model):
    # optionally resume from a checkpoint
    print("=> loading checkpoint")
    checkpoint = torch.load(name_prefix_for_saved_model + '-%d' % epoch)
    network.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {%d})" % epoch)
    return network, optimizer


def load_network_from_checkpoint(network, epoch, name_prefix_for_saved_model, stage=None, loss_function_name=''):
    # optionally resume from a checkpoint
    print("=> loading checkpoint '{}'")
    if stage != None:
        checkpoint = torch.load(name_prefix_for_saved_model + '-%d-%d%s' % (epoch, stage, loss_function_name))
    else:
        checkpoint = torch.load(name_prefix_for_saved_model + '-%d' % epoch)
    network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{%s}' (epoch {%d}) stage = %d" % (name_prefix_for_saved_model, epoch, stage))
    return network


def get_labels_matrix(labels_list_1, labels_list_2):
    matrix = torch.from_numpy(np.zeros((len(labels_list_1), len(labels_list_2))))
    for i, labels1 in enumerate(labels_list_1):
        for j, labels2 in enumerate(labels_list_2):
            # print('labels1 ', labels1)
            # print('labels2 ', labels2)
            new_labels_1 = np.delete(labels1.numpy(), np.where(labels1.numpy() == 0)[0])
            new_labels_2 = np.delete(labels2.numpy(), np.where(labels2.numpy() == 0)[0])
            # print('new_labels1 ', new_labels_1)
            # print('new_labels2 ', new_labels_2)
            # print('np.in1d(labels1, labels2) ', np.in1d(new_labels_1, new_labels_2))
            # print('np.in1d(labels1, labels2).any() ', np.in1d(new_labels_1, new_labels_2).any())
            # print('int(np.in1d(labels1.numpy(), labels2.numpy()).any()) ', int(np.in1d(new_labels_1, new_labels_2).any()))
            # input()
            matrix[i, j] = int(np.in1d(new_labels_1, new_labels_2).any())
    # print('matrix ', matrix)
    return matrix