import os
import torch
import torch.nn as nn
from collections import namedtuple
from torchnet.meter import ConfusionMeter, AverageValueMeter

from utils import tonumpy, totensor
from .model.utils.create_tool import Anchor_Target_Creator, Proposal_Target_Creator


LossTuple = namedtuple('LossTuple', ['rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss'])

class Faster_RCNN_Trainer(nn.Module):

    def __init__(self, opt, faster_rcnn):
        super(Faster_RCNN_Trainer, self).__init__()
        
        self.dev = torch.device("cuda: {}".format(opt.gpu) if torch.cuda.is_available() else "cpu")

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

        self.anchor_target_creator = Anchor_Target_Creator()
        self.proposal_target_creator = Proposal_Target_Creator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        # indicators for training status (confusion matrix)
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(11) 
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}


    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, h, w = imgs.shape
        img_size = (h, w)

        # backbone-vgg (features extractor)
        features = self.faster_rcnn.extractor(imgs)

        # region proposal network
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, tonumpy(bbox), tonumpy(label), self.loc_normalize_mean, self.loc_normalize_std)

        sample_roi_index = torch.zeros(len(sample_roi)) # because batch_size=1
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, sample_roi_index)

        # RPN loss
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(tonumpy(bbox), anchor, img_size)
        gt_rpn_label = totensor(gt_rpn_label).long()
        gt_rpn_loc = totensor(gt_rpn_loc)

        rpn_loc_loss = 0
        




    def _smooth_l1_loss(self, x, t, in_weight, sigma):
        sigma2 = sigma ** 2
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) +
            (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()
    

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        '''localizaion loss about only positive example'''

        in_weight = torch.zeros(gt_loc.shape).to(self.dev)
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).to(self.dev)] = 1
        loc_loss = self._smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
        loc_loss /= ((gt_label >= 0).sum().float())
        return loc_loss