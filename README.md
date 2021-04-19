# Retinanet-face_mask_detection
pytorch implementation of Retinanet for face mask detection (pretrained `ResNet50` backbone)<br>
more information at [base Retinanet codes](https://github.com/yhenon/pytorch-retinanet)<br>
## Results
Currently, this repo achieves 92.6% mAP at face_mask val dataset after 80 epochs trainig with a Resnet50 backbone(see `mAP_record.txt`) <br>
Further training for 140 epochs but got no improvement, both pre-trained model provided<br>

![image](https://github.com/simpletask1/Retinanet-face_mask_detection/tree/master/images/detect_0.jpg)
![image](https://github.com/simpletask1/Retinanet-face_mask_detection/tree/master/images/detect_1.jpg)<br>
![image](https://github.com/simpletask1/Retinanet-face_mask_detection/tree/master/images/detect_2.jpg)
![image](https://github.com/simpletask1/Retinanet-face_mask_detection/tree/master/images/detect_3.jpg)
## face_mask dataset
* 链接：https://pan.baidu.com/s/1pKBrwGAc-h3HO-36wAFEFA <br>
* 提取码：qpaz<br>
## Pre-trained model
Pre-trained models is available at:<br>
* 链接：https://pan.baidu.com/s/11WEDRXtzm3dZlwJWPaB4Yg <br>
* 提取码：nmgb<br> 

The state dict model can be loaded using:<br>
```Python
checkpoint = torch.load(PATH_TO_WEIGHTS)
retinanet.load_state_dict(checkpoint['model_state_dict']) # first need to remove prefix 'module.'
```
  
## Visualize
for images,use `detect.py`:<br>
```shell
python detect.py --model <path/to/model.pt> --img_dir <path/to/img folder> --save_path <path/to/save> 
```
for video,use `video.py`:<br>
```shell
python video.py --model <path/to/model.pt> --video <path/to/video or camera(default)> --save_path <path/to/save> 
```
