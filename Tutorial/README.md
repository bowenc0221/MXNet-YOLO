# Darknet framework documentation
This "tutorial" is created as a documentation for darknet framework. Most variables do not have clear meaning.

* [Darknet basic usage](#Basic-usage) : a tutorial on how to use darknet for yolo (train/test/val)  
  * [Compile](##Compile-darknet) : compile darknet  
  * [Architecture](##Darknet-architecture) : locate code  
  * [Train](##Train) : workflow for training  
  * [YOLO](##Run-YOLO-V2) : run yolo v2 on darknet  
  * [flags](##Optional-flags) : flags for running yolo v2  
* [Darknet cfg files](#cfg-files) : a tutorial on understanding darknet config files  
* [Darknet data structure](#data-structure) : data structure for saving network params and layer outpus

# Basic usage
## Compile darknet
1. modify Makefile
* set GPU=1 if you want to use gpus
* set CUDNN=1 if you want to use cudnn
* set OPENCV=1 if you want to use opencv to process images
2. ```make -j8```
3. try ```./darknet```. if it prints ```usage: ./darknet <function>``` then you are fine

## Darknet architecture
./darknet is from examples/darknet.c  
./darknet <function> will call other functions in examples/*.c  
For example: ```./darknet detector``` calls run_detector in examples/detector.c  

## Train
1. parse network
* parse config files and call make-xxx-layers to make corresponding layer according to network cfg files. Also print network structure to display.
2. load data
* For training:
  * DETECTION_DATA
* For testing:
  * LETTERBOX_DATA
    * load image using opencv
* multiscale train needs resize network.
3. train network
* update net.seen
* set net.train = 1
* forward/backward for subdivisions times
* update network
4. save weights
* in weight files, the first 4 elemets are:
  * int major = 0
  * int minor = 2
  * int revision = 0
  * size_t net.seen

## Run YOLO V2
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
## Optional flags
#### for ./darknet
1. ```-nogpu``` - do not use gpu
2. ```-i int``` - specify gpu index
#### for detector
1. ```-prefix str``` - prefix
2. ```-thresh float``` - thresh for test
3. ```-hier float``` - hier_thresh for test
4. ```-c int``` - cam_index for demo
5. ```-s int``` - frame_skip for demo
6. ```-avg int``` - for demo
7. ```-gpus str``` - string of gpu indexes seperated by comma. indicating how many gpus to use. e.g. -gpu 0,1,2,3 uses 4 gpus
8. ```-out str``` - outfile for test, detection image
9. ```-clear``` - for train, whether to clear network (set seen=0 I guess)
10. ```-fullscreen``` - for demo
11. ```-width int``` - for demo, resize image if >0
12. ```-height int``` - for demo
13. ```-fps int``` - for demo
# cfg files
see [config.md](https://github.com/bowenc0221/mxnet-yolo/blob/master/Tutorial/config.md)  
# data structure
see [data_structure.md](https://github.com/bowenc0221/mxnet-yolo/blob/master/Tutorial/data_structure.md)
