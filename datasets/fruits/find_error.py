import os
import xml.etree.ElementTree as et

'''find size of images = 0 * 0 file (error)'''

for f_name in os.listdir('./train'):
    if '.xml' in f_name:
        anno_path = os.path.join('./train', f_name)
        anno_tree = et.parse(anno_path)
        anno_root = anno_tree.getroot()

        w = int(anno_root.find("size").find("width").text)
        h = int(anno_root.find("size").find("height").text)

        if w*h == 0:
            print(anno_path)

for f_name in os.listdir('./test'):
    if '.xml' in f_name:
        anno_path = os.path.join('./test', f_name)
        anno_tree = et.parse(anno_path)
        anno_root = anno_tree.getroot()

        w = int(anno_root.find("size").find("width").text)
        h = int(anno_root.find("size").find("height").text)

        if w*h == 0:
            print(anno_path)

