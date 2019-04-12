#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import numpy as np
import math
from converter.layer import *


class AbstractConvolutionTF(object):
    __metaclass__ = abc.ABCMeta
    
    def parse_convolution(self, tf_operation, tf_session):
        self.weights = tf_session.run(get_tensors_with_weights(tf_operation)[0])        
        self.kernel_rows, self.kernel_columns, self.components_input, self.components_output = self.weights.shape        
        self.stride_rows, self.stride_columns = map(int, tf_operation.node_def.attr['strides'].list.i[1:3])
        padding_str = tf_operation.node_def.attr['padding'].s.decode()
        if padding_str == 'SAME':
            in_height, in_width = map(int, tf_operation.inputs[0].shape[1:3])
            out_height = math.ceil(float(in_height) / float(self.stride_rows))
            out_width = math.ceil(float(in_width) / float(self.stride_columns))
            self.padding_rows = max((out_height - 1) * self.stride_rows + self.kernel_rows - in_height, 0)
            self.padding_columns = max((out_width - 1) * self.stride_columns + self.kernel_columns - in_width, 0)            
        elif padding_str == 'VALID':
            self.padding_rows = 0
            self.padding_columns = 0
        else:
            raise ValueError('Not padding type: {}'.format(padding_str))
        self.rotate_weights()        
    
    @abc.abstractmethod    
    def rotate_weights(self):
        pass
        
    
class ConvolutionTF(Convolution, AbstractConvolutionTF):
    def parse(self, tf_operation, tf_session):
        super(ConvolutionTF, self).parse_convolution(tf_operation, tf_session)
    
    def rotate_weights(self):        
        self.weights = np.transpose(self.weights, (3,0,1,2)) # ColMajor


class PointwiseConvolutionTF(PointwiseConvolution, AbstractConvolutionTF):
    def parse(self, tf_operation, tf_session):
        super(PointwiseConvolutionTF, self).parse_convolution(tf_operation, tf_session)
    
    def rotate_weights(self):
        self.weights = np.transpose(self.weights, (3,2,0,1)) # ColMajor

    
class DepthwiseConvolutionTF(DepthwiseConvolution, AbstractConvolutionTF):
    def parse(self, tf_operation, tf_session):
        super(DepthwiseConvolutionTF, self).parse_convolution(tf_operation, tf_session)

    def rotate_weights(self):
        self.weights = np.transpose(self.weights, (3, 0, 1, 2)) # ColMajor        
        #self.weights = np.transpose(self.weights, (2, 3, 0, 1)) # RowMajor        
        

class BiasTF(Bias):
    def parse(self, tf_operation, tf_session):
        self.weights = tf_session.run(get_tensors_with_weights(tf_operation)[0])
        self.components = self.weights.shape[0]
        
        
class BatchNormTF(BatchNorm):
    def __init__(self):
        super(BatchNorm, self).__init__()
        
    def parse(self, tf_operation, tf_session):
        input_tensors = get_tensors_with_weights(tf_operation)
        beta_tensor = next((x for x in input_tensors if x.name.endswith('beta/read:0')))
        moving_mean_tensor = next((x for x in input_tensors if x.name.endswith('moving_mean/read:0')))
        gamma_tensor = next((x for x in input_tensors if x.name.endswith('gamma/read:0')))            
        moving_variance_tensor = next((x for x in input_tensors if x.name.endswith('moving_variance/read:0')))

        self.beta_weights = tf_session.run(beta_tensor)
        self.moving_mean_weights = tf_session.run(moving_mean_tensor)
        self.gamma_weights = tf_session.run(gamma_tensor)            
        self.moving_variance_weights = tf_session.run(moving_variance_tensor)
        self.epsilon = tf_operation.node_def.attr['epsilon'].f
        self.components = self.beta_weights.shape[0]

        
class ReluTF(Relu):    
    def parse(self, tf_operation, tf_session):
        self.components = tf_operation.outputs[0].shape[-1].value
        
class Relu6TF(Relu6):
    def parse(self, tf_operation, tf_session):
        self.components = tf_operation.outputs[0].shape[-1].value
    
class SoftmaxTF(Softmax):
    def parse(self, tf_operation, tf_session):
        self.components = tf_operation.outputs[0].shape[-1].value
    
class AvgPoolingTF(AvgPooling):
    def parse(self, tf_operation, tf_session):
        self.components = tf_operation.outputs[0].shape[-1].value
        self.kernel_rows, self.kernel_columns = map(int,tf_operation.node_def.attr['ksize'].list.i[1:3])
        self.stride_rows, self.stride_columns = map(int, tf_operation.node_def.attr['strides'].list.i[1:3])

class MaxPoolingTF(MaxPooling):
    def parse(self, tf_operation, tf_session):
        self.components = tf_operation.outputs[0].shape[-1].value
        self.kernel_rows, self.kernel_columns = map(int,tf_operation.node_def.attr['ksize'].list.i[1:3])
        self.stride_rows, self.stride_columns = map(int, tf_operation.node_def.attr['strides'].list.i[1:3])

def get_tensors_with_weights(tf_operation):
    return [op_input for op_input in tf_operation.inputs if op_input.name.endswith('read:0')]         

def get_layer_shape(tf_operation):
    tensor = get_tensors_with_weights(tf_operation)[0]
    return [dim.value for dim in tensor.get_shape()]

    
class LayerTensorflowFactory(LayerFactory):
    def create_layer(self, tf_operation):
        if tf_operation.type == 'Conv2D':
            kernel = get_layer_shape(tf_operation)[:2]
            if kernel == [1,1]:
                return PointwiseConvolutionTF()
            else:
                return ConvolutionTF()                
        elif tf_operation.type == 'DepthwiseConv2dNative': return DepthwiseConvolutionTF()
        elif tf_operation.type == 'FusedBatchNorm': return BatchNormTF()
        elif tf_operation.type == 'BiasAdd': return BiasTF()                
        elif tf_operation.type == 'Relu': return ReluTF()
        elif tf_operation.type == 'Relu6': return Relu6TF()
        elif tf_operation.type == 'AvgPool': return AvgPoolingTF()
        elif tf_operation.type == 'MaxPool': return MaxPoolingTF()        
        elif tf_operation.type == 'Softmax': return SoftmaxTF()                     
        else: return None
