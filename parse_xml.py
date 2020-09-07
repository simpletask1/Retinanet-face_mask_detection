import xml.etree.ElementTree as ET
import os

# xmls' path
source = 'D:/program file/condaProjects/1-datasets/maskDetection2/val/'
# images' path
goal = 'val/'
dir = os.listdir(source)
imgs = []
xmls = []
for name in dir:
    if name[-3:] == 'jpg':
        imgs.append(name)
    else:
        xmls.append(name)

with open('val_annotations.txt', 'a', encoding='utf-8') as f1:
    for i in range(len(xmls)):
        path = goal + imgs[i]
        tree = ET.parse(source + xmls[i])
        rect = {}
        line = ""
        root = tree.getroot()

        for ob in root.iter('object'):
            for cls in ob.iter('name'):
                c = cls.text
                if c == 'face_nask':
                    c = 'face_mask'

            for bndbox in ob.iter('bndbox'):
                for xmin in bndbox.iter('xmin'):
                    rect['xmin'] = xmin.text
                for ymin in bndbox.iter('ymin'):
                    rect['ymin'] = ymin.text
                for xmax in bndbox.iter('xmax'):
                    rect['xmax'] = xmax.text
                for ymax in bndbox.iter('ymax'):
                    rect['ymax'] = ymax.text

                line = path + ',' + rect['xmin'] + ',' + rect['ymin'] + ',' + rect['xmax'] + ',' + rect['ymax'] + "," + c + '\n'
                f1.write(line)

