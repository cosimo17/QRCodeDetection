[中文](README_zh.md)
# QRCode Detection
Deep learning based QRCode detection.

## Introduction
This is a project which depends on deep learning algorithm for QRCode detection.  
We have achieved fast and high-precision detection by using a yolov3-like detecter.  

Feature:  

+ Fast detection, more than 190 fps on GTX 1060.
+ High precision  
  Evaluate result on validation data 
  |Precision|Recall|Mean IOU|
  |  ----  | ----  |----|
  |0.987|0.819|0.798|  
  
+ Free deployment

## Installation
Please enable python in your machine.
```shell
git clone https://github.com/cosimo17/QRCodeDetection.git
cd QRCodeDetection
pip install -r requirements.txt

```
## Test
To test with the pretrained model, please download the pretrained weight file from [here](https://drive.google.com/file/d/1lqlQySkYehgkVJjZtRnYAICla7qSnxeG/view?usp=sharing).
```shell
python3 test.py \
	-w yolo_qrcode.h5 \
	-i test_images/1.jpg \
	-o ./result_1.jpg
```

## Training
* Before start training, please check [How to prepare dataset](data_generator/README.md)
* Run the kmean algorithm to generate priori anchor boxes
```shell
python3 utils/kmean.py \
		--root_dir your_dataset_dir \
		-n 6
```

Execute following command to start training:
```shell
python3 train.py \
	-d your_dataset_dir \
	-b 64 \
	-e 80
```
You can run ```python3 train.py --help``` to get help.  
During training, you can use tensorboard to visualize the loss curve.
```shell
tensorboard --logdir=./logs
```
![loss](assets/loss_curve.png)  

## Evaluate
Execute following command to evaluate the model performance:
```shell
python3 evaluate.py \
	-d your_dataset_dir \
	-b 64 \
	--score_threshold 0.5 \
	--iou_threshold 0.5 \
	-w yolo_qrcode.h5
```

## TODO
- [ ] Integrate decode module  
- [ ] Support docker container  
- [ ] Support openvino  
- [ ] Support tensorrt  
- [ ] Support tflite
