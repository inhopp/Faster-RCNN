import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from torchnet.meter import ConfusionMeter, AverageValueMeter

from utils import tonumpy, totensor, scalar
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


    def forward(self, imgs, bboxes, labels):
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, h, w = imgs.shape
        img_size = (h, w)

        # backbone-vgg (features extractor)
        features = self.faster_rcnn.extractor(imgs)

        # region proposal network
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size)

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
        gt_rpn_label = totensor(gt_rpn_label, self.dev).long()
        gt_rpn_loc = totensor(gt_rpn_loc, self.dev)

        rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.to(self.dev), ignore_index=-1)

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(totensor(_rpn_score, torch.device("cpu")), _gt_rpn_label.data.long())

        # Fast RCNN loss
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long.to(self.dev), totensor(gt_roi_label).long()]
        gt_roi_label = totensor(gt_roi_label).long()
        gt_roi_loc = totensor(gt_roi_loc)

        roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(self.dev))
        self.roi_cm.add(totensor(roi_score, torch.device("cpu")), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    
    def train_step(self, imgs, bboxes, labels):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)

        return losses


    def save(self, save_optimizer=False, save_path=None):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            save_path = './checkpoints/faster_rcnn_scratch_checkpoints.pth'

        os.makedirs(os.path.join(self.opt.ckpt_root, self.opt.data_name), exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, self.opt.data_name, "best_epoch.pt")
       
        torch.save(save_dict, save_path)

        return save_path


    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)

        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])

        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self

        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            
        return self



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


    def update_meters(self, losses):
        loss_d = {k: scalar(v) for k, v in losses._asdict().tems()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}