import torch
import numpy as np
import cv2
import skimage

import argparse
import os
import time

from retinanet import model
from retinanet.dataloader import ImgResizer, ImgNormalizer
from train import remove_prefix


def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple images detecting script.')

    parser.add_argument('--checkpoint', default='weights/csv_retinanet50_120epochs.pt',
                        help='Path to pre-trained checkpoint')
    parser.add_argument('--img_folder', default=None, help='Path to image folder contains images to be detected')
    parser.add_argument('--save_path', default=None,
                        help='Path to save detected images')

    parser = parser.parse_args(args)

    checkpoint = torch.load(parser.checkpoint)
    retinanet = model.resnet50(num_classes=2, pretrained=True)
    pretrained_dict = remove_prefix(checkpoint['model_state_dict'], 'module.')
    retinanet.load_state_dict(pretrained_dict)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    normalize = ImgNormalizer()
    resize = ImgResizer()

    img_source = parser.img_folder
    save_path = parser.save_path
    dir = os.listdir(img_source)
    for name in dir:
        img = cv2.imread(img_source + name)
        infer_img = normalize(img)
        infer_img = resize(infer_img)
        infer_img = infer_img.unsqueeze(0)

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(infer_img.cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(infer_img.float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                if int(classification[idxs[0][j]]) == 0:
                    label_name = 'face'
                else:
                    label_name = 'face_mask'
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)

            cv2.imwrite(save_path + 'det_{}'.format(name), img)
            # cv2.imshow('det_{}'.format(name), img)
            # cv2.waitKey(1500)
            # cv2.destroyWindow('det_{}'.format(name))


if __name__ == '__main__':
    main()
