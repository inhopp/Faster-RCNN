import os
import csv
import torch
import xml.etree.ElementTree as et
from PIL import Image
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, opt, phase, transform=None):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.img_size = opt.input_size
        self.transform = transform

        self.img_names = list()
        with open(os.path.join(self.data_dir, self.data_name, '{}.csv'.format(phase))) as f:
            reader = csv.reader(f)
            for line in reader:
                self.img_names.append(line)

        self.label2num = {}
        with open(os.path.join(self.data_dir, self.data_name, 'label.txt'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.label2num[line.strip()] = i


    def __getitem__(self, index):
        # read image
        img = Image.open(os.path.join(self.data_dir, self.data_name, self.img_names[index][0]))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # read annotation (xml file)
        anno_path = os.path.join(self.data_dir, self.data_name, self.img_names[index][0].replace('jpg', 'xml'))
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
            labels.append(self.label2num[obj_name])

            # bound boxes
            bbox = [obj_xmin, obj_ymin, obj_xmax, obj_ymax]
            boxes.append(bbox)


        # resize bbox
        W = int(anno_root.find("size").find("width").text)
        H = int(anno_root.find("size").find("height").text)

        W_ratio = self.img_size/W
        H_ratio = self.img_size/H
        ratio_list = [W_ratio, H_ratio, W_ratio, H_ratio]
        resized_boxes = []

        for box in boxes:
            box = list(map(float, box))
            bbox = [int(a*b) for a, b in zip(box, ratio_list)]
            resized_boxes.append(bbox)

        resized_boxes = torch.tensor(resized_boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return img, resized_boxes, labels


    def __len__(self):
        return len(self.img_names)