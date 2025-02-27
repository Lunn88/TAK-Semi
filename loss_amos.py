import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GADice(nn.Module):
    def __init__(self, GA=True):
        self.GA = GA
        super(GADice, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, cls, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10

        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union

        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        self.n_classes = inputs.size()[1]
        if len(inputs.size()) == len(target.size()) + 1:
            target = target.unsqueeze(1)

        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict & target shape do not match'
        smooth = 1e-10
        loss = 0.0
        for i in range(0, self.n_classes):
            if torch.sum(target[:, i]) > 0:
                dice_loss = self._dice_loss(i, inputs[:, i], target[:, i], weighted_pixel_map)
            else:
                if self.GA:
                    beta = inputs[:, i] / (torch.sum(1 - target[:, i]))
                    dice_loss = torch.sum(beta.detach() * inputs[:, i])
            loss += dice_loss * weight[i]

        return loss / self.n_classes


class GACE(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, k=10, gama=0.5):
        self.k = k
        self.gama = gama
        super(GACE, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target.long()
        self.n_classes = inp.size()[1]

        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, self.n_classes)
        target = target.view(-1, )
        res = super(GACE, self).forward(inp, target)

        n_instance = np.prod(res.shape)
        res, indices = torch.topk(res.view((-1,)), int(n_instance * self.k / 100), sorted=False)
        target = torch.gather(target, 0, indices)
        assert res.size() == target.size(), 'predict & target shape do not match'

        bg_w = np.power(int(n_instance * self.k / 100), self.gama)
        loss = 0.0
        smooth = 1e-10
        for i in range(0, self.n_classes):
            target_cls = (target == i).float()
            w = torch.pow(torch.sum(target_cls) + smooth, 1 - self.gama) * bg_w
            loss_cls = torch.sum(res * target_cls) / (w + smooth)
            loss += loss_cls

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self,
                 ignore_index=-1,
                 prompt_num=2):
        super(ContrastiveLoss, self).__init__()

        self.ignore_label = ignore_index
        self.prompt_num = prompt_num

    def sample_anchor(self, features, label, n_view):
        batch_size, feat_dim = features.shape[0], features.shape[-1]

        classes = []
        total_classes = 0
        for i in range(batch_size):
            i_label = label[i]

            i_classes = torch.unique(i_label)
            i_classes = [x for x in i_classes if x != self.ignore_label]

            classes.append(i_classes)
            total_classes += len(i_classes)
        if total_classes == 0:
            return None, None

        anc_features = []
        anc_labels = []

        for i in range(batch_size):
            i_label = label[i]
            i_classes = classes[i]

            for cls_id in i_classes:
                indices = (cls_id == i_label).nonzero()
                if indices.shape[0] <= n_view:
                    anc_features.append(features[i][indices].squeeze(1))
                else:
                    keep = torch.randperm(indices.shape[0])[:n_view]
                    indices = indices[keep]
                    anc_features.append(features[i][indices].squeeze(1))
                anc_labels.append(torch.full((indices.shape[0],), cls_id))

        anc_features = torch.cat(anc_features, dim=0)
        anc_labels = torch.cat(anc_labels).to(features.device)

        return anc_features, anc_labels

    def contrastive_loss(self, anchor, target):
        mask = torch.eq(target.unsqueeze(1), target.unsqueeze(0)).float().to(anchor.device)
        sim = torch.div(torch.matmul(anchor, anchor.T), 0.07)
        # temperature = self.temperature_list[target].unsqueeze(1)
        # sim = torch.div(torch.matmul(anchor, anchor.T), temperature)

        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        neg_mask = 1 - mask

        logits_mask = torch.ones((mask.shape[0], mask.shape[0]), dtype=bool).to(anchor.device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(dim=1, keepdim=True)
        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        loss = -mean_log_prob_pos.mean()

        return loss

    def forward(self, features_embedding_all, label, text_embedding_all, entropy_map, threshold, sample_num = 10):
        loss = 0.0
        # n_view_list = [40, 25, 10]
        n_view_list = [4, 2, 1]
        n_view_list = [int(x * sample_num) for x in n_view_list]

        mask = entropy_map > threshold
        pseudo_label = label[6:]
        pseudo_label[mask] = -1
        label = torch.cat([label[:6], pseudo_label], dim=0)

        for features, text_embedding, n_view in zip(features_embedding_all, text_embedding_all, n_view_list):
            i_label = label.float()
            i_label = F.interpolate(i_label, (features.shape[2], features.shape[3], features.shape[4]), mode='nearest')
            i_label = i_label.long()
            assert i_label.shape[-1] == features.shape[-1], '{} {}'.format(i_label.shape, features.shape)

            batch_size = features.shape[0]
            i_label = i_label.view(batch_size, -1)
            features = features.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, features.shape[1])

            anchor, target = self.sample_anchor(features, i_label, n_view)
            anchor = torch.cat([anchor, text_embedding], dim=0)
            anchor = F.normalize(anchor, p=2, dim=1)
            text_embedding_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                 9, 10, 11, 12, 13, 14, 15]).repeat(self.prompt_num).to(target.device)
            target = torch.cat([target, text_embedding_label], dim=0)
            loss += self.contrastive_loss(anchor, target)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10
        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i], weighted_pixel_map)
            # class_wise_dice.append(dice_loss)
            loss += dice_loss * weight[i]

        return loss / self.n_classes
