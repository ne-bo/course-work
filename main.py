import cifar
import learning
import test
import net
import params


def main():

    train_loader, test_loader = cifar.download_CIFAR100()

    network = net.Net(params.num_classes).cuda()

    learning.learning_process(train_loader=train_loader, network=network)
    
    test.test(test_loader=test_loader, network=network)


if __name__ == '__main__':
    main()
