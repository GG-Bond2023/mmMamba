import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.losses.utils import weighted_loss

@weighted_loss
def FreespaceLoss(prediction, target):
    freespace_loss = nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='none')# Returns non-reduced loss values (reduction='none'), dimensionality reduction is handled by the weighted_loss decorator
    return freespace_loss

@MODELS.register_module()
class loss_seg(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, loss_name='loss_seg',**kwargs):
        super(loss_seg, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
    def forward(self, prediction, target, weight=None, avg_factor=None,**kwargs):
        batch_size = prediction.size(0)
        prediction = prediction.contiguous().flatten()
        target = target.contiguous().flatten()
        loss_seg = self.loss_weight * FreespaceLoss(prediction, target, weight, reduction=self.reduction, avg_factor=avg_factor)
        loss_seg *= batch_size
        return loss_seg
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

@MODELS.register_module()
class loss_classif_reg(nn.Module):
    def __init__(self, classification, regression, loss_weight_classif=1.0, loss_weight_reg=1.0,  loss_name='loss_classif_reg',**kwargs):
        super(loss_classif_reg, self).__init__()
        self.param = {'classification':classification, 'regression':regression}
        self.loss_weight_classif = loss_weight_classif
        self.loss_weight_reg = loss_weight_reg
        self._loss_name = loss_name
    def forward(self, prediction, target, avg_factor=None,**kwargs):
        classif_loss, reg_loss = pixor_loss(prediction, target, param=self.param)
        classif_loss *= self.loss_weight_classif
        reg_loss *= self.loss_weight_reg
        # return classif_loss + reg_loss
        return [classif_loss, reg_loss]

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
    on difficult foreground detections.
    """

    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1 - prediction)

        # compute focal loss
        loss = -1 * (1 - pt) ** self.gamma * torch.log(pt + 1e-6)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def pixor_loss(batch_predictions, batch_labels, param):

    #########################
    #  classification loss  #
    #########################
    classification_prediction = batch_predictions[:, 0, :, :].contiguous().flatten()
    classification_label = batch_labels[:, 0, :, :].contiguous().flatten()

    if (param['classification'].lower() == 'focalloss'):
        focal_loss = FocalLoss(gamma=2)
        classification_loss = focal_loss(classification_prediction, classification_label)
    else:
        classification_loss = F.binary_cross_entropy(classification_prediction.double(),
                                                     classification_label.double(), reduction='sum')

    #####################
    #  Regression loss  #
    #####################

    regression_prediction = batch_predictions.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0) *
                                                                     regression_prediction.size(
                                                                         1) * regression_prediction.size(2),
                                                                     regression_prediction.size(3)])
    regression_label = batch_labels.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_label = regression_label.contiguous().view([regression_label.size(0) * regression_label.size(1) *
                                                           regression_label.size(2), regression_label.size(3)])

    positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
    pos_regression_label = regression_label[positive_mask.squeeze(), :]
    pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]

    T = batch_labels[:, 1:]
    P = batch_predictions[:, 1:]
    M = batch_labels[:, 0].unsqueeze(1)

    if (param['regression'].lower() == 'smoothl1loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
    else:
        reg_loss_fct = nn.L1Loss(reduction='sum')

    regression_loss = reg_loss_fct(P * M, T)
    NbPts = M.sum()
    if (NbPts > 0):
        regression_loss /= NbPts

    return classification_loss, regression_loss
