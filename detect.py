import torch
import numpy as np
import argparse
import time
import os
import csv
import cv2

from retinanet import model
from train import remove_prefix


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def prep_img(img):
    rows, cols, cns = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = img.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = img.astype(np.float32)
    img = new_image.astype(np.float32)
    img /= 255
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))

    return img, scale


def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    checkpoint = torch.load(model_path)
    retinanet = model.resnet50(num_classes=2, pretrained=True)
    pretrained_dict = remove_prefix(checkpoint['model_state_dict'], 'module.')
    retinanet.load_state_dict(pretrained_dict)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        image, scale = prep_img(image_orig)

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print('image shape:{},origin shape:{}, scale:{}'.format(image.shape, image_orig.shape, scale))
            scores, classification, transformed_anchors = retinanet(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(label_name)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                if label_name == 'face':
                    cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                else:
                    cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            cv2.imwrite(parser.save_path + 'detect_{}'.format(img_name), image_orig)
            # cv2.imshow('detections', image_orig)
            # cv2.waitKey(0)
            # cv2.destroyWindow('detections')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', default='test/',
                        help='Path to directory containing images')
    parser.add_argument('--save_path', default='results/',
                        help='Path to save detected images')
    parser.add_argument('--model', default='weights/csv_retinanet50_140epochs.pt',
                        help='Path to model')
    parser.add_argument('--class_list', default='class2id.csv',
                        help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model, parser.class_list)
