import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AllPairs(nn.Module):
    def __init__(self, in_features, out_features, l1_initialization=False, cuda=True):
        super(AllPairs, self).__init__()
        self.l1_initialization = l1_initialization
        self.in_features = in_features
        self.out_features = out_features


        if cuda:
            self.fc1 = nn.Linear(self.in_features, self.in_features).cuda()
            self.fc2 = nn.Linear(self.in_features, self.in_features).cuda()
            self.fc3 = nn.Linear(self.in_features, self.out_features).cuda()
        else:
            self.fc1 = nn.Linear(self.in_features, self.in_features)
            self.fc2 = nn.Linear(self.in_features, self.in_features)
            self.fc3 = nn.Linear(self.in_features, self.out_features)


        if self.l1_initialization:
            self.fc1.weight.data = torch.from_numpy(np.eye(self.fc1.weight.size(0))).float()
            self.fc1.bias.data.fill_(0.0)
            self.fc2.weight.data = torch.from_numpy(-np.eye(self.fc2.weight.size(0))).float()
            self.fc2.bias.data.fill_(0.0)
            self.fc3.weight.data.fill_(1.0)#/(self.fc3.weight.size(0) * self.fc3.weight.size(1)))
            #print('in all pairs self.fc3.weight.data ', self.fc3.weight.data)
            self.fc3.bias.data.fill_(0.0)

    def forward(self, input):
        #print('input', input)
        # split the input to 2 parts corresponding to 2 different batches
        batch_size = input.size(0)//2
        input_1 = self.fc1(input[:batch_size])
        input_2 = self.fc2(input[batch_size:])

        #print('input_1 ', input_1)
        #print('input_2 ', input_2)
        input_1 = input_1.expand(input_1.size(0), input_1.size(0), input_1.size(1))
        #print('input_1 after the expansion ', input_1)
        #print('input_1 difference between layers', input_1[:, 0, :] - input_1[:, 1, :])

        input_2 = torch.transpose(input_2.expand(input_2.size(0), input_2.size(0), input_2.size(1)), 0, 1)
        #print('input_2 after the expansion and transposition', input_2)
        #print('input_1 + input_2 ', input_1 + input_2)
        all_sums = input_1 + input_2
        #print('all_sums ', all_sums[:, :, 0])
        all_sums = all_sums.view(-1, self.in_features)

        #print('self.fc3 ',self.fc3)
        #print('self.fc3(all_sums) ', self.fc3(all_sums))
        return self.fc3(all_sums)


class EffectiveSimilarityNetwork(nn.Module):
    def __init__(self, number_of_input_features, number_of_output_neurons=1, l1_initialization=False, cuda=True):
        super(EffectiveSimilarityNetwork, self).__init__()

        ##################################################################
        #
        # Network parameters
        #
        ##################################################################

        # Learning Non-Metric Visual Similarity for Image Retrieval
        # https://arxiv.org/abs/1709.01353
        # Here we have architecture B from the original article

        self.number_of_input_features = number_of_input_features

        # parameters for first fully connected layer
        self.number_of_hidden_neurons_for_1_fully_connected = 2048

        # parameters for second fully connected layer
        self.number_of_hidden_neurons_for_2_fully_connected = 2048

        self.number_of_output_neurons = number_of_output_neurons

        ##################################################################
        #
        # Network structure
        #
        ##################################################################
        self.fc1 = AllPairs(in_features=self.number_of_input_features,
                            out_features=self.number_of_hidden_neurons_for_1_fully_connected,
                            l1_initialization=l1_initialization,
                            cuda=cuda)

        self.fc2 = nn.Linear(self.number_of_hidden_neurons_for_1_fully_connected,
                             self.number_of_hidden_neurons_for_2_fully_connected)

        self.fc3 = nn.Linear(self.number_of_hidden_neurons_for_2_fully_connected,
                             self.number_of_output_neurons)
        if l1_initialization:
            self.fc2.weight.data = torch.from_numpy(np.eye(self.fc2.weight.size(0))).float()
            self.fc2.bias.data.fill_(0.0)
            self.fc3.weight.data.fill_(1.0)#/(self.fc3.weight.size(0) * self.fc3.weight.size(1)))
            self.fc3.bias.data.fill_(0.0)
            #print('after!\n')

            #print('self.fc2.weight ', self.fc2.weight)
            #print('self.fc3.weight ', self.fc3.weight)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        #print('x after the all pairs layer', x)
        x = F.relu(self.fc2(x))
        #reg = torch.nn.Dropout(p=0.5)
        #y = x
        #print('x after the first linear layer', x, ' ', y.sum())
        x = self.fc3(x)

        #print('x after fc3 ', x.view(params.batch_size_for_similarity, params.batch_size_for_similarity))
        return x

# def create_similarity_network(number_of_input_features):
#    return SimilarityNetwork(number_of_input_features)
