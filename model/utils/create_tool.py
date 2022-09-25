import numpy as np
import torch
from torchvision.ops import nms
from bbox_tool import loc2bbox
from model.utils.bbox_tool import bbox2loc, bbox_iou

def get_inside_index(anchor, h, w):
    '''Calc indicies of anchors which are located completely inside of the image'''
    
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


def unmap(data, count, index, fill=0):
    '''Unmap a subset of item (data) back to the original set of items (of size count)'''
    
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class Proposal_Creator:
    '''generate RoI (train: 2000, test: 300)'''

    def __init__(self, parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms      # num of roi before nms
        self.n_train_post_nms = n_train_post_nms    # num of roi after nms
        self.n_test_pre_nms = n_test_pre_nms        # num of roi before nms
        self.n_test_post_nms = n_test_post_nms      # num of roi after nms
        self.min_size = min_size

    def __call__(self, dev, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)

        # clip to image boundary
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0]) # np.clip(array, min, max)
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # remove boxes smaller than min_size
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # sort (proposal, score) pairs by score
        order = score.ravel().argsort()[::-1]

        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # apply nms
        keep = nms(
            torch.from_numpy(roi).to(dev),
            torch.from_numpy(score).to(dev),
            self.nms_thresh)

        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        
        roi = roi[keep.cpu().numpy()]

        return roi


class Anchor_Target_Creator(object):
    '''Labeling positive/negative/ignore sample and return ground_truth anchor box location'''
    
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        h, w = img_size
        n_anchor = len(anchor) # (32*32) feature map * 9 anchor box -> 9216
        inside_index = get_inside_index(anchor, h, w) # 9216 -> 2272
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = unmap(label, n_anchor, inside_index, fill=-1)
        loc = unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

        
    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is ignore
        label = np.empty((len(inside_index), ), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]

        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index)-n_pos), replace=False)
            label[disable_index] = -1
        
        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]

        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index)-n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious