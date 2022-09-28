import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from data import generate_loader
from option import get_option
from model.faster_rcnn import Faster_RCNN_VGG16
from trainer import Faster_RCNN_Trainer
from calc_mAP import eval_detection

def inference(opt):
    dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    
    data_loader = generate_loader('temp', opt)
    print("data load complete")

    model = Faster_RCNN_VGG16(opt).to(dev)
    print("model construct complete")

    trainer = Faster_RCNN_Trainer(opt, model).to(dev)

    if opt.pretrained:
        load_path = os.path.join(opt.ckpt_root, opt.data_name, "best_epoch.pt")
        trainer.load(load_path)
 
    bboxes = []
    labels = []
    scores = []

    index = 1
    for ii, (img, bbox_, label_) in enumerate(data_loader):
        sizes = img.shape[1:]
        sizes = [sizes[1], sizes[2]]        
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        pred_bboxes_, pred_labels_, pred_scores_ = model.predict(img, [sizes])
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

        bboxes.append(pred_bboxes)
        labels.append(pred_labels)
        scores.append(pred_scores)

        img = img.cpu().detach().numpy()
        img = np.squeeze(img, axis=0)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        invert_norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        invert_norm_image = invert_norm_image.astype(np.uint8)
        img_path = os.path.join("./temp_" + str(index) + ".jpg")
        cv2.imwrite(img_path, invert_norm_image)
        index += 1

    cv2.namedWindow('temp')
    for i in range(index-1):
        img_path = os.path.join("./temp_" + str(i+1) + ".jpg")
        result_img = cv2.imread(img_path)
        cv2.rectangle(result_img, (int(bboxes[0][0][0][0]), int(bboxes[0][0][0][1])), (int(bboxes[0][0][0][2]), int(bboxes[0][0][0][3])), color=(0, 255, 0), thickness=10)

        cv2.imshow('result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ =='__main__':
    opt = get_option()
    torch.manual_seed(opt.seed)
    inference(opt)