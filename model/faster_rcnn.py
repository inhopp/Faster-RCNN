import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vgg16
from torchvision.ops import nms
from torchvision.ops import RoIPool

from utils import tonumpy, weight_initialize, totensor, nograd
from .utils.bbox_tool import loc2bbox

from .region_proposal_network import Region_Proposal_Network

def decom_vgg16():
    '''load vgg16 & select required layers'''

    model = vgg16(pretrained=True)
    features = list(model.features)[:30] # remove maxpool layer
    classifier = list(model.classifier) 
    
    del classifier[6] # remove last layer
    
    # remove dropout layer
    del classifier[5]
    del classifier[2]
    
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class VGG16RoIHead(nn.Module):
    def __init__(self, dev, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()

        self.dev = dev
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4) # bbox regressor
        self.score = nn.Linear(4096, n_class) # classifier

        weight_initialize(self.cls_loc, 0, 0.001)
        weight_initialize(self.score, 0, 0.01)

        self.n_class = n_class # num of classes + 1 (background)
        self.roi_size = roi_size # H and W of the feature maps after RoI-pooling
        self.spatial_scale = spatial_scale # roi resize scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        ''' x : 4d batch image variables '''

        roi_indices = totensor(roi_indices, self.dev).float()
        rois = totensor(rois, self.dev).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        # yx -> xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        # roi pooling per image
        pool = self.roi(x, indices_and_rois)
        # flatten
        pool = pool.view(pool.size(0), -1)
        
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores


class Faster_RCNN(nn.Module):
    def __init__(self, opt, extractor, rpn, head):
        super(Faster_RCNN, self).__init__()
        self.dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay

        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    @property
    def n_class(self):
        '''Total number of classes including background'''
        return self.head.n_class

    def forward(self, x, scale=1.):
        '''For Prediction'''
        img_size = x.shape[2:]

        hidden = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(hidden, img_size, scale)
        roi_cls_locs, roi_scores = self.head(hidden, rois, roi_indices)

        return roi_cls_locs, roi_scores, rois, roi_indices

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()

        # skip cls_id = 0 (background)
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l-1) * np.ones((len(keep),)))
            score.append((prob_l[keep].cpu().numpy()))

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score


    @nograd
    def predict(self, imgs, sizes=None):
        self.eval()
        prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None], self.dev).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # Assume batch_size = 1
            roi_cls_loc = roi_cls_loc.data
            roi_score = roi_scores.data
            roi = totensor(rois, self.dev) / scale

            # convert bbox in image coordinate
            mean = torch.Tensor(self.loc_normalize_mean).to(self.dev).repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).to(self.dev).repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)), tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox, self.dev)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)

            # clip bbox
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[0])

            prob = (F.softmax(totensor(roi_score, self.dev), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.train()
        return bboxes, labels, scores


    def get_optimizer(self):
        lr = self.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': self.weight_decay}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer


    def scale_lr(self, decay=0.9):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer        


class Faster_RCNN_VGG16(Faster_RCNN):
    # downsample 16x for output of conv5 in vgg16
    feat_stride = 16

    def __init__(self, opt):
        dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        n_fg_class = opt.num_classes # except background

        extractor, classifier = decom_vgg16()

        rpn = Region_Proposal_Network(opt, 512, 512, feat_stride=self.feat_stride)
        head = VGG16RoIHead(dev, n_class=n_fg_class+1, roi_size=7, spatial_scale=(1./self.feat_stride), classifier=classifier)
        
        super(Faster_RCNN_VGG16, self).__init__(opt, extractor, rpn, head,)

