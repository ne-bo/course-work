from torch.autograd import Variable
import torch


def test(test_loader, network):
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = network(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))