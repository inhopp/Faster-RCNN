import numpy as np
import torch
from torchvision.ops import nms
from bbox_tool import loc2bbox

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