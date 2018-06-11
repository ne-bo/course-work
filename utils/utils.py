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


def get_labels_matrix_fast(labels_list_1, labels_list_2, negative_sign=0):
    recall_at_k = 0.0
    all_labels_1 = labels_list_1.cpu().numpy()
    all_labels_2 = labels_list_2.cpu().numpy()
    if negative_sign == 0:
        matrix = np.zeros((len(labels_list_1), len(labels_list_2)), dtype=int)
    if negative_sign == -1:
        matrix = -np.ones((len(labels_list_1), len(labels_list_2)), dtype=int)
    positives = 0
    for i, current_label in enumerate(all_labels_1):
        # print('current_label ', current_label)
        for label in current_label:
            # print('label ', label)
            if label != 0:  # if some of labes == 0 than we have <3 characters on the image
                matrix[i, np.where(all_labels_2 == label)[0]] = 1
    print('matrix fast ', matrix[:3, :3])
    # print('mpercentage of ones in matrix fast', np.sum(matrix)/(matrix.shape[0] * matrix.shape[1]))
    matrix = torch.from_numpy(matrix)
    return matrix


def get_layer_with_number(network, layer_number):
    return torch.nn.Sequential(*list(network.features.children())[:layer_number]).cuda()