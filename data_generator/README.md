[中文](README_zh.md)
# How to prepare training data

Yu can using generated fake data to train this model, or you can collect your own dataset for training.  
We suggest you train on fake data first, then finetune on your own dataset.

## Generate fake dataset
We provide two scripts for data generation.
* Generate QRCode image
```shell
mkdir qrcodes  
python3 data_generator/generate_qrcode.py \
		-n 1500 \
		-o qrcodes
```

* Prepare some background images. (Such as imagenet, open image.)

* Generate training data
```shell
python3 generate_training_data.py \
		-fg qrcodes \
		-bg your_dir \
		-o training_ds \
		-n 40000 \
		--shape 256
```
The generated data looks like following:  
![dataset](../assets/dataset.png)  
We have already generated 40000 images and labels. You can download them from here: [dataset](https://drive.google.com/file/d/1Mv9fC8e4-IJq3MLQ_QA846o4TTjn-9ui/view?usp=sharing)

## Prepare your own dataset
Of course, you can prepare your own dataset by yourself.
* Collect images data

* Annotate your images   
  You can use any tools you like to annotate your images. [Labelme](https://github.com/wkentaro/labelme) will be a good choice.
* Convert the label format 
  After annotation, you should convert the label format.  
    ```
  training_ds
  ------------
  |
  |---000001.jpg
  |---000001.txt 
  |---000002.jpg
  |---000002.txt
  |---...
  |---...
  |---...
  |---xxxxxx.jpg
  |---xxxxxx.txt
  For each image, there should be a txtfile which has the same name with image's name.
  The format of txt:
    cx,cy,w,h,1.0, 1.0
    cx,cyw,h,1.0,1.0
  Each line represents an qrcode object.
  cx,cy means the center coordinates, w,h means the width and the height. All coordinates are normalized to [0-1]