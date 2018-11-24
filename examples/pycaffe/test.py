# -*- coding: utf-8 -*-
# author = sai
from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto

def My_layer(source_dir, batch_size, target_size):
    param_str = "'source_dir': source_dir, 'batch_size': batch_size, 'target_size': target_size"
    mylayer = L.Python(module='pyreadlayer', layer='DataReadLayer', param_str=param_str)
    print(mylayer)
    to_proto(mylayer)

def make(filename):
    with open(filename, 'w') as f:
        print(My_layer(source_dir='/data/images/cifar-10/test', batch_size=50, target_size=100), file=f)

if __name__=='__main__':
    net = caffe.Net('myRead.prototxt', caffe.TEST)
    net.forward()
    images = net.blobs['images'].data
    print(images.shape)