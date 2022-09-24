import os
import csv
import xml.etree.ElementTree as et
from PIL import Image
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, opt, phase, transform=None):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.phase = phase
        self.transform = transform

        self.img_names = list()
        with open(os.path.join(self.data_dir, self.data_name, '{}.csv'.format(phase))) as f:
            reader = csv.reader(f)
            for line in reader:
                self.img_names.append(line)


def __getitem__(self, index):
    # read image
    img = Image.open(os.path.join(self.data_dir, self.data_name, self.img_names[index]))
    img = img.convert('RGB')

    if self.transform is not None:
        img = self.transform(img)

    # read annotation (xml file)
    anno_path = self.img_names[index].replace('jpg', 'xml')
    anno_tree = et.parse(anno_path)
    anno_root = anno_tree.getroot()

    labels = []
    boxes = []

    for node in anno_root.findall("object"):
        obj_name = node.find("name").text
        obj_xmin = node.find("bndbox").find("xmin").text
        obj_ymin = node.find("bndbox").find("ymin").text
        obj_xmax = node.find("bndbox").find("xmax").text
        obj_ymax = node.find("bndbox").find("ymax").text   

        # labels
        if obj_name == 'apple':
            labels.append(1)
        elif obj_name == 'banana':
            labels.append(2)
        else:
            labels.append(3)

        # bound boxes
        bbx = [obj_xmin, obj_ymin, obj_xmax, obj_ymax]
        boxes.append(bbx)


    return img, labels, boxes


def __len__(self):
    return len(self.data)