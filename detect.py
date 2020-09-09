import numpy as np
import os
import time
import argparse

import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer
from train import remove_prefix


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    if caption == 'face':
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    else:
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple detecting script')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', default='class2id.csv',
                        help='Path to file containing class list (see readme)')
    parser.add_argument('--model', default='weights/csv_retinanet50_140epochs.pt', help='Path to model (.pt) file.')
    parser.add_argument('--img_folder', default='test/',
                        help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--save_path', default='./',
                        help='Path to save detected images')

    parser = parser.parse_args(args)

    # randomly generate a csv file for detecting
    if os.path.exists('test.txt'):
        os.remove('test.txt')
    dir = os.listdir(parser.img_folder)
    with open('test.txt', 'a', encoding='utf-8') as f1:
        for i in range(len(dir)):
            path = parser.img_folder + dir[i]
            line = path + ',' + '0' + ',' + '0' + ',' + '10' + ',' + '10' + "," + 'face' + '\n'
            f1.write(line)
    os.rename('test.txt', 'test.csv')

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='train2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file='test.csv', class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    checkpoint = torch.load(parser.model)
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

    unnormalize = UnNormalizer()

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                if label_name == 'face':
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                draw_caption(img, (x1, y1, x2, y2), label_name)
                print(label_name)

            cv2.imwrite(parser.save_path + 'detect{}.jpg'.format(idx), img)


if __name__ == '__main__':
    main()
