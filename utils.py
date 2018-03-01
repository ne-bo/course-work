import matplotlib.pyplot as plt
import numpy as np
import torch
import shutil
import params

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


def load_network_from_checkpoint(network, epoch, name_prefix_for_saved_model, stage=None):
    # optionally resume from a checkpoint
    print("=> loading checkpoint '{}'")
    if stage != None:
        checkpoint = torch.load(name_prefix_for_saved_model + '-%d-%d' % (epoch, stage))
    else:
        checkpoint = torch.load(name_prefix_for_saved_model + '-%d' % epoch)
    network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {%d})" % epoch)
    return network
