import argparse
import xml.etree.ElementTree as ET
import os

parser = argparse.ArgumentParser(description='Build Annotations.')
parser.add_argument('dir', default='..', help='Annotations.')

# sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = [('2007', 'train'), ('2007', 'val')]

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}


def convert_annotation(year, image_id, f, data_dir):
    in_file = os.path.join(data_dir, 'VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        classes = list(classes_num.keys())
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))



if __name__=="__main__":
    for year, image_set in sets:
        print(year, image_set)
        if image_set=="test":
            data_dir = "VOCdevkit_test"
            #read_dir = os.path.join("VOCdevkit_test/VOC%s/ImageSets/Main/%s.txt" % (year, image_set))
            #write_path = os.path.join("VOCdevkit_test", '%s_%s.txt' % (year, image_set)) 
        else:
            data_dir = "VOCtrainval_06-Nov-2007/VOCdevkit"
            #read_path = os.path.join('VOCtrainval_06-Nov-2007/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set))
            #write_path = os.path.join("VOCtrainval_06-Nov-2007/VOCdevkit", '%s_%s.txt' % (year, image_set))
        with open(os.path.join(data_dir, "VOC%s/ImageSets/Main/%s.txt" % (year, image_set)), 'r') as f:
            image_ids = f.read().strip().split()
        with open(os.path.join(data_dir, '%s_%s.txt' % (year, image_set)), 'w') as f:
            for image_id in image_ids:
                f.write('%s/VOC%s/JPEGImages/%s.jpg' % (data_dir, year, image_id))
                convert_annotation(year, image_id, f, data_dir)
                f.write('\n')