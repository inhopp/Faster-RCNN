import os
import torch

from tqdm import tqdm

from data import generate_loader
from option import get_option
from model.faster_rcnn import Faster_RCNN_VGG16
from trainer import Faster_RCNN_Trainer
from calc_mAP import eval_detection

def train(opt):
    dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    print("device: ", dev)

    train_loader = generate_loader('train', opt)
    test_loader = generate_loader('test', opt)
    print("data load complete")

    model = Faster_RCNN_VGG16(opt).to(dev)
    print("model construct complete")

    trainer = Faster_RCNN_Trainer(opt, model).to(dev)

    if opt.pretrained:
        load_path = os.path.join(opt.ckpt_root, opt.data_name, "best_epoch.pt")
        trainer.load(load_path)
    
    lr = opt.lr
    lr_decay = opt.lr_decay
    
    
    best_mAP = 0

    for epoch in range(opt.n_epoch):
        trainer.reset_meters()

        for ii, (img, bbox_, label_) in enumerate(tqdm(train_loader)):
            img, bbox, label = img.to(dev).float(), bbox_.to(dev), label_.to(dev)
            trainer.train_step(img, bbox, label)

        eval_result = eval(test_loader, model)

        trainer.faster_rcnn.scale_lr(lr_decay)
        lr = lr * lr_decay

        if eval_result['map'] > best_mAP:
            best_mAP = eval_result['map']
            trainer.save(opt)


def eval(data_loader, faster_rcnn):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, gt_bboxes_, gt_labels_) in tqdm(enumerate(data_loader)):
        sizes = imgs.shape[1:]
        sizes = [sizes[1], sizes[2]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    result = eval_detection(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, use_07_metric=True)
    
    return result



if __name__ =='__main__':
    opt = get_option()
    torch.manual_seed(opt.seed)
    train(opt)