import torch
from torch.autograd import Variable


# Learning Deep Embeddings with Histogram Loss
# https://arxiv.org/abs/1611.00822


class HistogramLossForSimilarity(torch.nn.Module):
    # R - number of bins
    # delta - step size
    def __init__(self, R):
        super(HistogramLossForSimilarity, self).__init__()
        self.R = R
        self.delta = 2.0 / (float(self.R) - 1.0)
        self.t = torch.arange(-1, 1, self.delta).view(-1, 1).cuda()
        self.tsize = self.t.size()[0]
        self.t = self.t.cuda()

    def delta_ijr(self, i, j, r, similarities_matrix):
        if self.t[r - 1] <= similarities_matrix[i, j] <= self.t[r]:
            return (similarities_matrix[i, j] - self.t[r - 1]) / self.delta
        else:
            if self.t[r] <= similarities_matrix[i, j] <= self.t[r + 1]:
                return (self.t[r + 1] - similarities_matrix[i, j]) / self.delta
            else:
                return 0.0

    def delta_s_r(self, r, s):
        if self.t[r - 1] <= s <= self.t[r]:
            return (s - self.t[r - 1]) / self.delta
        else:
            if self.t[r] <= s <= self.t[r + 1]:
                return (self.t[r + 1] - s) / self.delta
            else:
                return 0.0

    def S_plus_minus(self, signs_matrix):
        mask = signs_matrix.eq(1)
        S_plus = torch.masked_select(signs_matrix, mask).numel()
        S_minus = signs_matrix.numel() - S_plus
        return S_plus, S_minus

    def h_r(self, similarities_matrix, signs_matrix, r, S, plus_or_minus='+'):
        if plus_or_minus == '+':
            mask = signs_matrix.eq(1)
        else:
            mask = signs_matrix.eq(-1)
        print('torch.masked_select(similarities_matrix, mask) ', torch.masked_select(similarities_matrix, mask))
        masked = torch.masked_select(similarities_matrix, mask)
        masked.cpu().apply_(self.delta_s_r)
        print('self.delta_s_r(r, torch.masked_select(similarities_matrix, mask)) ',
              self.delta_s_r(r, torch.masked_select(similarities_matrix, mask)))
        return torch.sum(self.delta_s_r(r, torch.masked_select(similarities_matrix, mask))) / S

    def phi_r_plus(self, similarities_matrix, signs_matrix, r, S_plus):
        summ = 0.0
        for q in range(r):
            summ = summ + self.h_r(similarities_matrix, signs_matrix, q, S_plus, plus_or_minus='+')
        return sum

    def forward_natasha(self, similarities_matrix, signs_matrix):
        S_plus, S_minus = self.S_plus_minus(signs_matrix)
        loss = 0.0
        for r in range(self.R):
            loss = loss + self.h_r(similarities_matrix, signs_matrix, r, S_minus, '-') * \
                   self.phi_r_plus(similarities_matrix, signs_matrix, r, S_plus)
        return loss

    def forward(self, similarities_matrix, signs_matrix):
        #print('features ', features)
        #print('classes ', np.sort(classes.data.cpu().numpy()))
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (delta_repeat == (self.t - self.delta)) & inds
            indsb = (delta_repeat == self.t) & inds
            s_repeat_[~(indsb | indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - Variable(self.t) + self.delta)[indsa] / self.delta
            s_repeat_[indsb] = (-s_repeat_ + Variable(self.t) + self.delta)[indsb] / self.delta
            return s_repeat_.sum(1) / size

        classes_eq = signs_matrix.data

        dists = similarities_matrix
        #print('dists = ', dists)
        #print('classes_eq ', classes_eq)
        s_inds = torch.triu(torch.ones(dists.size()), 1).byte()
        s_inds = s_inds.cuda()
        #print('s_inds', s_inds)
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum()
        neg_size = (~classes_eq[s_inds]).sum()

        #print('pos_size ', pos_size)
        #print('neg_size', neg_size)
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        delta_repeat = (torch.floor((s_repeat.data + 1) / self.delta) * self.delta - 1).float()

        histogram_pos = histogram(pos_inds, pos_size)
        histogram_neg = histogram(neg_inds, neg_size)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
        histogram_pos_inds = histogram_pos_inds.cuda()

        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)

        #print('histogram_neg ', histogram_neg )
        #print('histogram_pos_cdf ', histogram_pos_cdf)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss