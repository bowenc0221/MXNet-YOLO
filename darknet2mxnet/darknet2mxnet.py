import sys
import os
sys.path.append('/data/home/bcheng/git_darknet/caffe/python')
mxnet_path = os.path.join('/data/home/bcheng/git_point/mx_rfcn/external/mxnet', 'mxnet_point')
sys.path.insert(0, mxnet_path)
import mxnet as mx
# import caffe
import numpy as np
from collections import OrderedDict
from cfg import *
from prototxt import *

def darknet2mxnet(cfgfile, weightfile, prefix):
    blocks = parse_cfg(cfgfile)

    num_input = []
    fp = open(weightfile, 'rb')
    # header = np.fromfile(fp, count=4, dtype=np.int32)
    header = np.fromfile(fp, count=3, dtype=np.int32)
    seen = np.fromfile(fp, count=1, dtype=np.int64)
    buf = np.fromfile(fp, dtype=np.float32)
    fp.close()

    print buf.shape[0]
    start = 0
    layer_id = 1  # track conv layer
    arg_params = dict()
    aux_params = dict()
    total_layer_id = 1

    for block in blocks:
        if block['type'] == 'net':
            num_input.append(int(block['channels']))
        elif block['type'] == 'convolutional': # name, type, param(#input, #output, kernel, pad, stride)
            conv_layer = OrderedDict()
            if block.has_key('name'):
                conv_layer['name'] = block['name']
            else:
                conv_layer['name'] = 'layer%d-conv' % layer_id
            convolution_param = OrderedDict()
            convolution_param['num_input'] = num_input[-1]
            convolution_param['num_output'] = int(block['filters'])
            # update #input
            num_input.append(int(block['filters']))
            convolution_param['kernel_size'] = int(block['size'])
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            else:
                convolution_param['bias_term'] = 'true'
            conv_layer['convolution_param'] = convolution_param

            # print 'conv%d' % total_layer_id, num_input[-2], num_input[-1]

            conv_weight_name = conv_layer['name'] + '_weight'
            conv_bias_name = conv_layer['name'] + '_bias'
            arg_params[conv_weight_name] = np.zeros((conv_layer['convolution_param']['num_output'],
                                                     conv_layer['convolution_param']['num_input'],
                                                     conv_layer['convolution_param']['kernel_size'],
                                                     conv_layer['convolution_param']['kernel_size']))

            print conv_weight_name, arg_params[conv_weight_name].shape

            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                if block.has_key('name'):
                    bn_layer['name'] = '%s-bn' % block['name']
                else:
                    bn_layer['name'] = 'layer%d-bn' % layer_id

                bn_beta = bn_layer['name'] + '_beta'
                bn_gamma = bn_layer['name'] + '_gamma'
                bn_avg = bn_layer['name'] + '_moving_mean'
                bn_var = bn_layer['name'] + '_moving_var'
                arg_params[bn_beta] = np.zeros((conv_layer['convolution_param']['num_output']))
                arg_params[bn_gamma] = np.zeros((conv_layer['convolution_param']['num_output']))
                aux_params[bn_avg] = np.zeros((conv_layer['convolution_param']['num_output']))
                aux_params[bn_var] = np.zeros((conv_layer['convolution_param']['num_output']))

                print bn_gamma, arg_params[bn_gamma].shape

                start = load_conv_bn2caffe(buf, start, arg_params, aux_params, conv_weight_name,
                                           bn_beta, bn_gamma, bn_avg, bn_var)
            else:
                arg_params[conv_bias_name] = np.zeros((conv_layer['convolution_param']['num_output']))
                print conv_bias_name, arg_params[conv_bias_name].shape
                start = load_conv2caffe(buf, start, arg_params, conv_weight_name, conv_bias_name)

            layer_id = layer_id+1
            total_layer_id = total_layer_id + 1

        elif block['type'] == 'route':
            sub_idx = map(int, block['layers'].split(','))
            get_output = 0
            # print 'route%d' % total_layer_id
            for i in sub_idx:
                # print i, num_input[i]
                get_output = get_output + num_input[i]
            num_input.append(get_output)
            # print 'output', num_input[-1]
            total_layer_id = total_layer_id + 1

        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            num_input.append(num_input[-1]*stride*stride)
            # print 'reorg%d' % total_layer_id, num_input[-2], num_input[-1]
            total_layer_id = total_layer_id + 1

        elif block['type'] == 'maxpool':
            num_input.append(num_input[-1])
            # print 'max%d' % total_layer_id, num_input[-2], num_input[-1]
            total_layer_id = total_layer_id + 1

    print num_input
    print start
    save_checkpoint(prefix, 0, arg_params, aux_params)

def load_conv_bn2caffe(buf, start, arg_params ,aux_params, conv_weight_name, bn_beta, bn_gamma, bn_avg, bn_var):
    conv_weight = arg_params[conv_weight_name]
    running_mean = aux_params[bn_avg]
    running_var = aux_params[bn_var]
    scale_weight = arg_params[bn_gamma]
    scale_bias = arg_params[bn_beta]

    arg_params[bn_beta] = mx.nd.array(np.reshape(buf[start:start+scale_bias.size], scale_bias.shape))
    start = start + scale_bias.size
    arg_params[bn_gamma] = mx.nd.array(np.reshape(buf[start:start+scale_weight.size], scale_weight.shape))
    start = start + scale_weight.size
    aux_params[bn_avg] = mx.nd.array(np.reshape(buf[start:start+running_mean.size], running_mean.shape))
    start = start + running_mean.size
    aux_params[bn_var] = mx.nd.array(np.reshape(buf[start:start+running_var.size], running_var.shape))
    start = start + running_var.size
    # bn_param[2].data[...] = np.array([1.0])
    # convolution weight
    arg_params[conv_weight_name] = mx.nd.array(np.reshape(buf[start:start+conv_weight.size], conv_weight.shape))
    start = start + conv_weight.size
    return start

def load_conv2caffe(buf, start, arg_params, conv_weight_name, conv_bias_name):
    weight = arg_params[conv_weight_name]
    bias = arg_params[conv_bias_name]
    arg_params[conv_bias_name] = mx.nd.array(np.reshape(buf[start:start+bias.size], bias.shape))
    start = start + bias.size
    arg_params[conv_weight_name] = mx.nd.array(np.reshape(buf[start:start+weight.size], weight.shape))
    start = start + weight.size
    return start

def save_checkpoint(prefix, epoch, arg_params, aux_params):
    save_dict = {('arg:%s' % k) : v for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v for k, v in aux_params.items()})
    param_name = '%s-%04d.params' % (prefix, epoch)
    mx.nd.save(param_name, save_dict)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('Usage:')
        print('python darknet2mxnet.py darknet.cfg darknet.weights mxnet.params')
        print('')
        print('please add name field for each block to avoid generated name')
        exit()

    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    prefix = sys.argv[3]
    darknet2mxnet(cfgfile, weightfile, prefix)
