import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from utils import weight_initialize
from .utils.bbox_tool import generate_anchor
from .utils.create_tool import Proposal_Creator


class Region_Proposal_Network(nn.Module):
    def __init__(self, dev, in_channels=512, mid_channels=512, feat_stride=16, proposal_creator_params=dict()):
        super(Region_Proposal_Network, self).__init__()
        self.dev = dev
        self.feat_stride = feat_stride
        self.proposal_layer = Proposal_Creator(self, **proposal_creator_params)
        self.anchor_base = generate_anchor()
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        weight_initialize(self.conv1, 0, 0.01)
        weight_initialize(self.score, 0, 0.01)
        weight_initialize(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        # x : feature map extreaced from images -> (N, C, H, W)
        n, _, hh, ww = x.shape

        # (h*w*9) anchor box coordinates
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        
        hidden = F.relu(self.conv1(x))

        # bounding box offset & object socres
        rpn_locs = self.loc(hidden)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.socre(hidden)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()

        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        rpn_scores = rpn_scores.view(n, -1, 2)

        # generate proposal
        rois = list()
        roi_indices = list() # index of image

        for i in range(n):
            roi = self.proposal_layer(self.dev, rpn_locs[i].cpu().data.numpy(), rpn_fg_scores[i].cpu().data.numpy(), anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, index=0)
        roi_indices = np.concatenate(roi_indices, index=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    '''
    enumerate anchor boxes of all pixels
    (generate_anchor() : 9 anchor boxes on specific pixel)
    '''

    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.rabel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor