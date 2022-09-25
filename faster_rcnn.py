import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool
from utils import weight_initialize
from utils import totensor

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