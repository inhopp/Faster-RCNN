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


def generate_anchor():
    base_size = 16
    ratio = [0.5, 1, 2]
    scale = [8, 16, 32]

    anchor_boxes = np.zeros((len(ratio) * len(scale), 4), dtype=np.float32)

    x_center = base_size/2
    y_center = base_size/2

    for i in range(len(ratio)):
        for j in range(len(scale)):
            h = base_size * scale[j] * np.sqrt(ratio[i])
            w = base_size * scale[j] * np.sqrt(1./ratio[i])

            index = i * len(ratio) + j
            anchor_boxes[index, 0] = y_center - h / 2. # ymin
            anchor_boxes[index, 1] = x_center - w / 2. # xmin
            anchor_boxes[index, 2] = y_center + h / 2. # ymax
            anchor_boxes[index, 3] = x_center + w / 2. # xmax

    return anchor_boxes


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # To compare 1 bbox(a) and multiple bbox(b), increasing dimension using None
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2]) # top_left
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:]) # bottom_right

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    iou = area_i / (area_a[:, None] + area_b - area_i)

    return iou