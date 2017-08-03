# configuration files for darknet
Two types for config files, one for loading data and one for setting up network.  

* [Data config](#data-config)  
* [Network config](#network-cfg)  

# data config
## loading:  
src/option_list.c::read_data_cfg(filename)  
## keywords:  
### for train:  
"train" - file contains list of images for training  
"backup" - directory to save weights  
### for valid:  
"valid" - file contains list of images for validation  
"names" - file contains label names  
"results" - directory to save validation results  
"map" - used for softmax_tree (not sure what its exactly usage)  
"eval" - str, can be set to "voc" (default), "coco", "imagenet"  
### for test:  
"names" - file contains label names  
### for demo:  
"classes" - number of classes  
"names" - file contains label names  

# network cfg
## Data structure needed
### Section
#### defination (src/parser.c)
```C
typedef struct{
    char *type;
    list *options;
}section;
```
### List
#### defination
```C
typedef struct list{
    int size;
    node *front;
    node *back;
} list;
```
### Node
#### defination
```C
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;
```
## loading:
### Sample cfg file
```
[net]
batch=64
subdivisions=8
height=416
width=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 80200
policy=steps
steps=40000,60000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2
...

```
### train:  
src/network.c::load_network(cfg, weights, clear) = parse_network_cfg + load_weights  
### valid/test:  
/src/parser.c::parse_network_cfg(filename)  
## structure:
Read_cfg returns a list of sections of the network.  
In the cfg file, [ ] defines a section containing list. e.g. [convolution] defines a section of convolution layer  
Must start with [net] or [network]  
#### options for net:  
batch, learning_rate, momentum, decay, subdividions, time_steps, notruth, adam (with: B1, B2, eps), height, width, channels, inputs, max_crop, min_crop, center, angle, aspect, saturation, exposure, hue, policy, burn_in, power, max_batches  
policy= STEP(step, scale), STEPS(steps, scales), EXP(gamma), SIG(gamma, step), POLY or RANDOM()  

**_net.batch = batch / subdividions to reduce gpu memory._**
