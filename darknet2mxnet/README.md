# darknet2mxnet
convert darknet .weights files to mxnet .params files

# What you need
1. compile darknet following: https://pjreddie.com/darknet/install/
2. compile mxnet following: http://mxnet.io/get_started/install.html
3. make a copy of darknet2mxnet.py and put it into pytorch-caffe-darknet-convert
4. modify mxnet_path

# How to use
python darknet2mxnet.py (darknet network .cfg file) (darknet .weights file) (mxnet params prefix)
for example:
```
python darknet2mxnet.py yolo-voc.cfg yolo-voc.weights yolo-voc
```
and you will get yolo-voc-0000.params containing arg_params and aux_params dictionaries.
