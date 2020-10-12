# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = ['ResNeXt50', 'ResNeXt101', 'ResNeXt152']
"""
https://arxiv.org/pdf/1611.05431.pdf
https://github.com/facebookresearch/ResNeXt

Downsampling of conv3, 4, and 5 is done by stride-2 
convolutions in the 3×3 layer of the first block in each stage

ReLU is performed right after each BN, expect for the output of the 
block where ReLU is performed after the adding to the shortcut
"""

class BottleneckBlock(tf.keras.Model):
    def __init__(self, filters, strides, cardinality = 32):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters = filters,
                                            kernel_size = 1,
                                            strides = 1,
                                            groups = cardinality,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(filters = filters,
                                            kernel_size = 3,
                                            strides = strides,
                                            groups = cardinality,
                                            padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.conv3 = tf.keras.layers.Conv2D(filters = 2 * filters,
                                            kernel_size = 1,
                                            strides = 1,
                                            groups = cardinality,
                                            padding = 'same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.shortcut_conv1.ad = tf.keras.layers.Conv2D(filters = 2 * filters,
                                            kernel_size = 1,
                                            strides = strides,
                                            padding = 'same')   
        self.shortcut_bn1 = tf.keras.layers.BatchNormalization()
        
    def call(self, input_tensor, training = None):
                
        x = self.conv1(input_tensor)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training = training)
        x = tf.nn.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training = training)
        x = tf.nn.relu(x)
        
        shortcut = self.shortcut_conv1(input_tensor)
        shortcut = self.shortcut_bn1(shortcut, training = training)
        
        x = tf.keras.layers.add([shortcut, x])
        return tf.nn.relu(x)
    
def bottleneck_blocks(filters, strides, cardinality, n_blocks):
    res_block = tf.keras.Sequential()

    # We make sure always at leaste one block exists
    res_block.add(BottleneckBlock(filters = filters, 
                                  strides = strides, 
                                  cardinality = cardinality))
    for _ in range(1, n_blocks):
        res_block.add(BottleneckBlock(filters = filters, 
                                      strides = 1, 
                                      cardinality = cardinality))
    return res_block


class ResNeXt():
    def __init__(self, n_blocks, , cardinality, n_classes):
        super(ResNeXt, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = 7,
                                            strides = 2,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size = 3,
                                                  strides = 2,
                                                  padding = 'same')
        
        # This layers will be named conv as in the original paper
        self.conv2 = bottleneck_blocks(filters = 128,
                                       strides = 1,
                                       cardinality = 32,
                                       n_blocks = n_blocks[0])
        
        self.conv3 = bottleneck_blocks(filters = 256,
                                       strides = 2,
                                       cardinality = 32,
                                       n_blocks = n_blocks[1])
        
        self.conv4 = bottleneck_blocks(filters = 512,
                                       strides = 2,
                                       cardinality = 32,
                                       n_blocks = n_blocks[2])
        
        self.conv5 = bottleneck_blocks(filters = 1024,
                                       strides = 2,
                                       cardinality = 32,
                                       n_blocks = n_blocks[3])
        
        #its the same as GlobalAveragePool2D
        self.avgpool1 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units = n_classes,
                                         activation = tf.nn.softmax)
        
        
    def call(self, input_tensor, training = None, **kwargs):
        
        x = self.conv1(input_tensor)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x, training = training)
        x = self.conv3(x, training = training)
        x = self.conv4(x, training = training)
        x = self.conv5(x, training = training)
        
        x = self.avgpool1(x)
        # already includes softmax with the number of classes
        x = self.fc1(x)
        
        return x
        
def ResNeXt50(n_classes):
    return ResNeXt(n_blocks = [3, 4, 6, 3], cardinality = 32, n_classes = n_classes)

def ResNeXt101(n_classes):
    return ResNeXt(n_blocks = [3, 4, 23, 3], cardinality = 32, n_classes = n_classes)

def ResNeXt152(n_classes):
    return ResNeXt(n_blocks = [3, 8, 36, 3], cardinality = 32, n_classes = n_classes)
        