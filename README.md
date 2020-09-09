# Retinanet-face_mask_detection
pytorch implementation of Retinanet for face mask detection (version `ResNet50` pretrained)<br>
more information at [base Retinanet codes](https://github.com/yhenon/pytorch-retinanet)<br>
## Results
Currently, this repo achieves 92.6% mAP after 80 epochs trainig with a Resnet-50 backbone<br>
## face_mask dataset
链接：https://pan.baidu.com/s/1gQ1RrOiDfn94kNPO-mFYSA <br>
提取码：cmqf<br>
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
python detect.py --checkpoint <path/to/model.pt> --img_folder <path/to/img folder> --save_path <path/to/save> 
```
for video,use `video_demo.py`:<br>
```shell
python video_demo.py --checkpoint <path/to/model.pt> --video <path/to/video or camera> --save_path <path/to/save> 
```

