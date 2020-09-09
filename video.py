import argparse
import time
import csv

import torch
import numpy as np
import cv2

from retinanet import model
from train import remove_prefix
from detect import load_classes, draw_caption, prep_img


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple video detecting script.')

    parser.add_argument('--class_list', default='class2id.csv',
                        help='Path to CSV file listing class names (see README)')
    parser.add_argument('--model', default='weights/csv_retinanet50_140epochs.pt',
                        help='Path to pre-trained checkpoint')
    parser.add_argument('--video', default=0, help='Video file to run detection on')
    parser.add_argument('--save_path', default='results/detected_video000.avi',
                        help='Path to save detected video')

    parser = parser.parse_args(args)

    with open(parser.class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    print("Loading network.....")
    checkpoint = torch.load(parser.model)
    retinanet = model.resnet50(num_classes=2, pretrained=True)
    pretrained_dict = remove_prefix(checkpoint['model_state_dict'], 'module.')
    retinanet.load_state_dict(pretrained_dict)
    print("Network successfully loaded")

    CUDA = torch.cuda.is_available()
    use_gpu = True

    if use_gpu:
        if CUDA:
            retinanet = retinanet.cuda()

    if CUDA:
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    cap = cv2.VideoCapture(parser.video, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(0)  # for webcam

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 3
    ret, frame = cap.read()
    videoWriter = cv2.VideoWriter(parser.save_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            infer_img, scale = prep_img(frame)

            if CUDA:
                infer_img = torch.from_numpy(infer_img)
                infer_img = infer_img.cuda()

            with torch.no_grad():
                if CUDA:
                    scores, classification, transformed_anchors = retinanet(infer_img.cuda().float())
                else:
                    scores, classification, transformed_anchors = retinanet(infer_img.float())

            frames += 1
            print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                score = scores[j]

                caption = '{} {:.3f}'.format(label_name, score)
                draw_caption(frame, (x1, y1, x2, y2), caption)

                if label_name == 'face':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                print(label_name)

            cv2.imshow("frame", frame)
            videoWriter.write(frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print('Elapsed time: {}'.format(time.time() - start))
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            videoWriter.release()  # release at the end of loop
            break


if __name__ == '__main__':
    main()
