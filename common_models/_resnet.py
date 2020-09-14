# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = ['ResNet18', 'ResNet34']

# Implementation of ResNet
# https://arxiv.org/pdf/1512.03385.pdf

class IdentityBlock(tf.keras.Model):
    """
    Creates normal single block for ResNets 18 and 34
    """

    def __init__(self, filters, strides):
        super(IdentityBlock, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters = filters,
                                            kernel_size = 3,
                                            strides = strides,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        
        self.conv2 = tf.keras.layers.Conv2D(filters = filters,
                                            kernel_size = 3,
                                            strides = 1,
                                            padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        # When the shortcuts go across feature maps of two sizes, 
        # they are performed with a stride of 2
        
        # In this case we apply a (1,1) conv so the dimensions coincide
        if strides != 1:
            self.residual = tf.keras.Sequential()
            self.residual.add(tf.keras.layers.Conv2D(filters = filters,
                                                     kernel_size = 1,
                                                     strides = strides))

            self.residual.add(tf.keras.layers.BatchNormalization())
        else:
            self.residual = lambda x: x
        
    def call(self, input_tensor, training = None):

        x_residual = self.residual(input_tensor)

        x = self.conv1(input_tensor)
        x = self.bn1(x, training = training)
        # using the keras activations may cause issues in custom layers
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training = training)

        x = tf.keras.layers.add([x_residual, x])    
        return tf.nn.relu(x)


def residual_blocks(filters, strides, n_blocks):
    """
    Creates a sequence of residual blocks with 
    the same properties
    """
    res_block = tf.keras.Sequential()
    res_block.add(IdentityBlock(filters = filters,
                                strides = strides))
    
    for _ in range(1, n_blocks):
        res_block.add(IdentityBlock(filters = filters,
                                    strides = 1))
    return res_block

class ResNet(tf.keras.Model):
    """ResNet Base Model.
        Can be used to build both ResNet-18 and ResNet-34
    """
    def __init__(self, n_res_blocks, n_classes):
        super(ResNet, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = 7,
                                            strides = 2,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size = 3,
                                                    strides = 2,
                                                    padding = 'same')
        
        self.res_layer1 = residual_blocks(filters = 64,
                                          strides = 1,
                                          n_blocks = n_res_blocks[0])
        
        self.res_layer2 = residual_blocks(filters = 128,
                                          strides = 2,
                                          n_blocks = n_res_blocks[1])
        
        self.res_layer3 = residual_blocks(filters = 256,
                                          strides = 2,
                                          n_blocks = n_res_blocks[2])
        
        self.res_layer4 = residual_blocks(filters = 512,
                                          strides = 2,
                                          n_blocks = n_res_blocks[3])
        
        self.avgpool1 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units = n_classes,
                                        activation = tf.nn.softmax)
        
    def call(self, input_tensor, training = None):

        x = self.conv1(input_tensor)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.maxpool1(x)

        x = self.res_layer1(x, training = training)
        x = self.res_layer2(x, training = training)
        x = self.res_layer3(x, training = training)
        x = self.res_layer4(x, training = training)

        x = self.avgpool1(x)
        x = self.fc(x)
            
        return x

# 18 and 34 Layer ResNet have the same architecture 
def ResNet18(n_classes):
    return ResNet(n_res_blocks = [2, 2, 2, 2], n_classes = n_classes)

def ResNet34(n_classes):
    return ResNet(n_res_blocks = [3, 4, 6, 3], n_classes = n_classes)