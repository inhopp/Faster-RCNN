import numpy as np


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