import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA
from torch.autograd import Variable

from datasets.loaders import UKB
from evaluation import test
from networks_and_layers.l2_normalization import L2Normalization
from utils import params


def learn_PCA_matrix_for_spocs(spocs, desired_dimension):
    # print('spocs in learn PCA ', spocs.data.shape )
    U, S, V = torch.svd(torch.t(spocs.data))
    print('U.shape ', U.shape)
    print('S.shape ', S.shape)
    print('V.shape ', V.shape)
    return U[:, :desired_dimension], S[:desired_dimension]

def learn_PCA_matrix_for_spocs_with_sklearn(spocs, desired_dimension):
    print('spocs in learn PCA ', spocs.shape)
    pca = PCA(n_components=desired_dimension)
    U, S, V = pca._fit(torch.t(spocs).cpu().numpy())
    print('U ', U.shape)
    print('S ', S.shape)
    print('V ', V.shape)
    print('pca.components_.shape', pca.components_.shape)
    return U[:, :desired_dimension], S[:desired_dimension]


# outputs is a Tensor with the shape batch_size x 512 x 37 x 37
# we should return the Tensor of size batch_size x 256
def compute_spoc_by_outputs(outputs):
    batch_size = outputs.size(0)
    desired_representation_length = outputs.size(1)
    # sum pooling
    sum_pooled = torch.sum(outputs.view(batch_size, desired_representation_length, -1), dim=2)
    # L2 - normalization
    normalization = L2Normalization()
    spocs = normalization(sum_pooled)
    return spocs




def save_all_spocs_and_labels(test_loader, network, file_spoc, file_labels, test_or_train):
    all_spocs = torch.cuda.FloatTensor()
    all_labels = torch.LongTensor()
    progress = 0
    for data in test_loader:
        progress = progress + 1
        if progress % 100 == 0:
            print('progress ', progress)
        images, labels = data
        #print('labels in batch ', labels),
        outputs = network(Variable(images).cuda())

        spocs = compute_spoc_by_outputs(outputs)
        #print('spocs ', spocs)
        all_spocs = torch.cat((all_spocs, spocs.data), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
        #print('all_spocs ', all_spocs)
    print('all_spocs', all_spocs)
    print('all_labels', all_labels)
    torch.save(all_spocs, file_spoc)
    torch.save(all_labels, file_labels)
    return all_spocs, all_labels


def read_spocs_and_labels(file_spoc, file_labels):
    all_spocs = torch.load(file_spoc)
    all_labels = torch.load(file_labels)
    print('all_spocs', all_spocs)
    print('all_labels', all_labels)
    return all_spocs, all_labels


def get_spoc():
    #train_loader, test_loader = birds.download_BIRDS_for_representation(data_folder='CUB_200_2011')

    train_loader, test_loader = UKB.download_UKB_for_representation(data_folder='ukbench/full')


    # this magic code allows us to take the network up to the specific layer even if this layer has no it's own name
    # here 19 is a number of the desired level in the initial pretrained network
    vgg = models.vgg16(pretrained=True)
    print('full vgg ', vgg)
    representation_network = nn.Sequential(*list(vgg.features.children())[:29]).cuda()

    # batch_size x 512 x 18 x 18 should be batch_size x 512 x 37 x 37
    print('next(representation_network ', representation_network)
    all_spocs_train, all_labels_train = save_all_spocs_and_labels(train_loader, representation_network,
                                                      'all_spocs_file_train', 'all_labels_file_train', 'train')

    all_spocs_test, all_labels_test = save_all_spocs_and_labels(test_loader, representation_network,
                                                      'all_spocs_file_test', 'all_labels_file_test', 'test')

    all_spocs_train, all_labels_train = read_spocs_and_labels('all_spocs_file_train', 'all_labels_file_train')
    all_spocs_test, all_labels_test = read_spocs_and_labels('all_spocs_file_test', 'all_labels_file_test')

    # PCA
    PCA_matrix, singular_values = learn_PCA_matrix_for_spocs(all_spocs_train, 256)
    torch.save(PCA_matrix, 'PCA_matrix')
    torch.save(singular_values, 'singular_values')

    all_spocs_train = torch.div(torch.mm(all_spocs_train, PCA_matrix), singular_values)
    all_spocs_test = torch.div(torch.mm(all_spocs_test, PCA_matrix), singular_values)

    print('all_spocs_train_after_pca', all_spocs_train)

    # L2 - normalization
    normalization = L2Normalization()
    all_spocs_train = normalization(Variable(all_spocs_train)).data
    all_spocs_test = normalization(Variable(all_spocs_test)).data

    torch.save(all_spocs_train, 'all_spocs_file_train_after_pca')
    torch.save(all_spocs_test, 'all_spocs_file_test_after_pca')

    print("Evaluation on train")
    test.full_test_for_representation(k=params.k_for_recall,
                                      all_outputs=all_spocs_train, all_labels=all_labels_train)
    print("Evaluation on test")
    test.full_test_for_representation(k=params.k_for_recall,
                                      all_outputs=all_spocs_test, all_labels=all_labels_test)


# get_spoc()