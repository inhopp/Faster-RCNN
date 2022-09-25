import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import nms

def generate_anchor():
    base_size = 16
    ratio = [0.5, 1, 2]
    scale = [8, 16, 32]

    anchor_boxes = np.zeros((len(ratio) * len(scale), 4), dtype=np.float32)

    x_center = base_size/2
    y_center = base_size/2

    for i in len(ratio):
        for j in len(scale):
            h = base_size * scale[j] * np.sqrt(ratio[i])
            w = base_size * scale[j] * np.sqrt(1./ratio[i])

            index = i * len(ratio) + j
            anchor_boxes[index, 0] = y_center - h / 2. # ymin
            anchor_boxes[index, 1] = x_center - w / 2. # xmin
            anchor_boxes[index, 2] = y_center + h / 2. # ymax
            anchor_boxes[index, 3] = x_center + w / 2. # xmax

    return anchor_boxes


def loc2bbox(src_bbox, loc):
    '''anchor boxes on pixel_location'''

    if src_bbox.shape[0] == 0:
        return np.zeros((0,4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_y_center = src_bbox[:, 0] + 0.5 * src_height
    src_x_center = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    y_center = dy * src_height[:, np.newaxis] + src_y_center[:, np.newaxis]
    x_center = dx * src_width[:, np.newaxis] + src_x_center[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = y_center - 0.5 * h
    dst_bbox[:, 1::4] = x_center - 0.5 * w
    dst_bbox[:, 2::4] = y_center + 0.5 * h
    dst_bbox[:, 3::4] = x_center + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    '''predicted anchor box + destination bbox (ground_truth box) -> location'''

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    y_center = src_bbox[:, 0] + 0.5 * height
    x_center = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_y_center = dst_bbox[:, 0] + 0.5 * base_height
    base_x_center = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_y_center - y_center) / height
    dx = (base_x_center - x_center) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


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


def weight_initialize(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
        

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