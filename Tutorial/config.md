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
## loading:
train:  
src/network.c::load_network(cfg, weights, clear) = parse_network_cfg + load_weights  
valid/test:  
/src/parser.c::parse_network_cfg(filename)  
## structure:
[ ] defines a section  

