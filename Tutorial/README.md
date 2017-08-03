# Darknet framework documentation
This "tutorial" is created as a documentation for darknet framework. Most variables do not have clear meaning.

# Basic usage
## compile darknet
1. modify Makefile
* set GPU=1 if you want to use gpus
* set CUDNN=1 if you want to use cudnn
* set OPENCV=1 if you want to use opencv to process images
2. "make -j8"
3. try "./darknet". if it prints "usage: ./darknet <function>" then you are fine

## darknet architecture
./darknet is from examples/darknet.c

./darknet <function> will call other functions in examples/*.c
  
For example: ./darknet detector calls run_detector in examples/detector.c

## run yolo v2
train (on voc: darknet19 as backbone network):
```
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
test (on single image: data/dog.jpg):
```
./darknet detector test cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_final.weights data/dog.jpg
```
validation (on the entire test set):
```
./darknet detector valid cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_final.weights
```
validation with flip (on the entire test set):
```
./darknet detector valid2 cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_final.weights
```

## cfg files
see config.md
