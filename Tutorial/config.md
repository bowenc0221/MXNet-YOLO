# configuration files for darknet
Two types for config files, one for loading data and one for setting up network.  

* [Data config](#data-config)  
* [Network config](#network-cfg)  

# data config
### loading:  
src/option_list.c::read_data_cfg(filename)  
### keywords:  
#### for train:  
"train" - file contains list of images for training  
"backup" - directory to save weights  
#### for valid:  
"valid" - file contains list of images for validation  
"names" - file contains label names  
"results" - directory to save validation results  
"map" - used for softmax_tree (not sure what its exactly usage)  
"eval" - str, can be set to "voc" (default), "coco", "imagenet"  
#### for test:  
"names" - file contains label names  
#### for demo:  
"classes" - number of classes  
"names" - file contains label names  

# network cfg
### loading:
