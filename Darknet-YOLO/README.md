# Darknet-YOLO
add functions to save weights and feature maps  
seprate conv output and BN output  

## usage
1. put detector.c in darknet/examples  
2. put convolutional_layer.c in darknet/src
3. compile darknet
4. use the following command
```
./darknet detector -nogpu feature cfg/voc.data cfg/yolo-voc.cfg backup/yolo-voc_final.weights VOCdevkit/VOC2007/JPEGImages/000001.jpg -layer 0
```
this command will save params and feature map of first layer (index starts from 0)  
params are saved to params/  
feature maps are saved to feature/  
params and features can be load in python using
```Python
import numpy as np
weights = np.fromfile(WEIGHTS_FILE, np.float32)
```
