import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGAttN(nn.Module):
    """Base class for Graph Attention Networks. Provides shared loss and metric utilities."""

    @staticmethod
    def loss(logits, labels, nb_classes, class_weights):
        # class_weights: [nb_classes] tensor
        sample_wts = (F.one_hot(labels, nb_classes).float() * class_weights).sum(dim=-1)
        xentropy = F.cross_entropy(logits, labels, reduction='none')
        return (xentropy * sample_wts).mean()

    @staticmethod
    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = F.cross_entropy(logits, labels.argmax(dim=-1), reduction='none')
        mask = mask.float()
        mask = mask / mask.mean()
        loss = loss * mask
        return loss.mean()

    @staticmethod
    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Sigmoid cross-entropy loss with masking."""
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = loss.mean(dim=1)
        mask = mask.float()
        mask = mask / mask.mean()
        loss = loss * mask
        return loss.mean()

    @staticmethod
    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct = logits.argmax(dim=1).eq(labels.argmax(dim=1)).float()
        mask = mask.float()
        mask = mask / mask.mean()
        correct = correct * mask
        return correct.mean()

    @staticmethod
    def micro_f1(logits, labels, mask):
        """Micro F1 score with masking."""
        predicted = torch.round(torch.sigmoid(logits)).int()
        labels = labels.int()
        mask = mask.int().unsqueeze(-1)

        tp = (predicted * labels * mask).count_nonzero()
        fp = (predicted * (labels - 1) * mask).count_nonzero()
        fn = ((predicted - 1) * labels * mask).count_nonzero()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        fmeasure = (2 * precision * recall) / (precision + recall + 1e-9)
        return fmeasure.float()

    @staticmethod
    def preshape(logits, labels, nb_classes):
        log_resh = logits.reshape(-1, nb_classes)
        lab_resh = labels.reshape(-1)
        return log_resh, lab_resh

    @staticmethod
    def confmat(logits, labels):
        preds = logits.argmax(dim=1)
        nb_classes = logits.shape[1]
        cm = torch.zeros(nb_classes, nb_classes, dtype=torch.long, device=logits.device)
        for t, p in zip(labels, preds):
            cm[t, p] += 1
        return cm
