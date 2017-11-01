import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import cifar
import params
import torch.optim as optim
import learning
import utils
from torch.optim import lr_scheduler


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SmallResnet(nn.Module):
    def __init__(self, block, layers, num_classes):
        #print('inside network init')
        self.inplanes = 16
        super(SmallResnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #?????

        self.layer1 = self._make_layer(block=block, planes=16, blocks=layers[0])
        self.layer2 = self._make_layer(block=block, planes=32, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=64, blocks=layers[2], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.last_bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes,
                          out_channels=planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(inplanes=self.inplanes,
                        planes=planes,
                        stride=stride,
                        downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)#????

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.last_bn(x)# added in order to make histogramm loss work

        return x


def small_resnet_for_cifar(num_classes, n):
    """Constructs a small Small_ResNet for cifar as in 4.2 part in https://arxiv.org/pdf/1512.03385.pdf

       num_classes - number of classes in classification task
       n - number of pairs of layers in each of 3 big layer

       Quote:
           The plain/residual architectures follow the form in Fig. 3 (middle/right).
           The network inputs are 32×32 images, with the per-pixel mean subtracted.

           The first layer is 3×3 convolutions.
           Then we use a stack of 6n layers with 3×3 convolutions
                on the feature maps of sizes {32, 16, 8} respectively,
                with 2n layers for each feature map size.
                The numbers of filters are {16, 32, 64} respectively.
                The subsampling is performed by convolutions with a stride of 2.
           The network ends with a global average pooling,
           a 10-way fully-connected layer,
           and softmax.

           There are totally 6n+2 stacked weighted layers.
    """
    model = SmallResnet(block=BasicBlock,
                        layers=[n, n, n],
                        num_classes=num_classes)
    return model


def test_on_cifar_10():
    train_loader, test_loader = cifar.download_CIFAR10(batch_size=128)

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

    network = small_resnet_for_cifar(num_classes=10, n=3).cuda()

    restore_epoch = 0
    optimizer = optim.SGD(network.parameters(),
                          lr=0.1,
                          weight_decay=0.0001,
                          momentum=0.9)
    multi_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[82, 123],
                                                  gamma=0.1)

    ##################################################################
    #
    # Optional recovering from the saved file
    #
    ##################################################################
    # restore_epoch = 11
    # network, optimizer = utils.load_network_and_optimizer_from_checkpoint(network=network,
    #                                                                      optimizer=optimizer,
    #                                                                      epoch=restore_epoch)

    ##################################################################
    #
    # Pre-training for classification
    #
    ##################################################################

    learning.learning_process(train_loader=train_loader,
                              network=network,
                              criterion=nn.CrossEntropyLoss(),
                              test_loader=test_loader,
                              mode=params.mode_classification,
                              optimizer=optimizer,
                              start_epoch=restore_epoch,
                              lr_scheduler=multi_lr_scheduler)


#test_on_cifar_10()
