#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import numpy as np
import math

def arguments(attribute, arg1, arg2):
    if (arg1 == arg2) or (arg2 == None):
        return '%s<%d>' % (attribute, arg1)
    return '%s<%d, %d>' % (attribute, arg1, arg2)

def components(input_size, output_size = None):
    return arguments('components', input_size, output_size)

def kernel(rows, columns = None):
    return arguments('kernel', rows, columns)

def stride(rows, columns = None):
    return arguments('stride', rows, columns)

def padding(rows, columns = None):
    return arguments('padding', rows, columns)

class Layer(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod    
    def has_weights(self):
        pass
    
    @abc.abstractmethod    
    def type_name(self):
        pass
    
    @abc.abstractmethod
    def parse(self):
        pass
    
    @abc.abstractmethod    
    def check_completeness(self):
        pass
    
    @abc.abstractmethod    
    def to_model_entry(self):
        pass
     
    
class NotWeightedLayer(Layer):
    __metaclass__ = abc.ABCMeta
    
    def has_weights(self):
        return False
    

class Relu(NotWeightedLayer):
    def type_name(self):
        return 'relu'

    def to_model_entry(self):
        return '%s<float, %s>()' % (self.type_name(), components(self.components))
    
    def check_completeness(self):
        assert hasattr(self, 'components') and self.components is not None
    
class Relu6(Relu):
    def to_model_entry(self):
        return '%s<float, %s>(0, 6)' % (self.type_name(), components(self.components))
    
class AvgPooling(NotWeightedLayer):
    def type_name(self):    
        return 'avg_pooling'

    def to_model_entry(self):
        return '%s<float, %s, %s, %s>()' % (self.type_name(), components(self.components), kernel(self.kernel_rows, self.kernel_columns), stride(self.stride_rows, self.stride_columns))
    
    def check_completeness(self):
        assert hasattr(self, 'components') and self.components is not None
        assert hasattr(self, 'kernel_rows') and self.kernel_rows is not None
        assert hasattr(self, 'kernel_columns') and self.kernel_columns is not None        
        assert hasattr(self, 'stride_rows') and self.stride_rows is not None
        assert hasattr(self, 'stride_columns') and self.stride_columns is not None

class MaxPooling(NotWeightedLayer):
    def type_name(self):    
        return 'max_pooling'

    def to_model_entry(self):
        return '%s<float, %s, %s, %s>()' % (self.type_name(), components(self.components), kernel(self.kernel_rows, self.kernel_columns), stride(self.stride_rows, self.stride_columns))
    
    def check_completeness(self):
        assert hasattr(self, 'components') and self.components is not None
        assert hasattr(self, 'kernel_rows') and self.kernel_rows is not None
        assert hasattr(self, 'kernel_columns') and self.kernel_columns is not None        
        assert hasattr(self, 'stride_rows') and self.stride_rows is not None
        assert hasattr(self, 'stride_columns') and self.stride_columns is not None    
    
    
class Softmax(NotWeightedLayer):
    def type_name(self):    
        return 'softmax'

    def to_model_entry(self):
        return '%s<float, %s>()' % (self.type_name(), components(self.components))

    def check_completeness(self):
        assert hasattr(self, 'components') and self.components is not None    
    
class WeightedLayer(Layer):
    __metaclass__ = abc.ABCMeta
        
    def has_weights(self):
        return True
    

    def dump_weights(self, layer_index, weights_dir_path):
        self.__dump_weights(self.type_name(), layer_index, self.weights, weights_dir_path)
        
    
    def __dump_weights(self, layer_type, layer_index, weights, weights_dir_path, layer_var_name = 'weight'):
        self.layer_index = layer_index
        
        # create file for the layer
        f = open(weights_dir_path + '/' + str(layer_index) + layer_type + '.h', 'w')

        #write the layer weights in the file
        f.write('#include <polimidl/layers/alignment.hpp>\n')
        f.write('alignas(polimidl::layers::buffer_alignment::byte_alignment) const float '+ layer_var_name + str(layer_index) + '[] = {')

        weights_formatted = ','.join(["%.32f" % val + 'f' for val in weights.flatten()])
        f.write(weights_formatted)
        f.write('}; \n')
        f.close()
            
    
class Convolution(WeightedLayer):
    
    @abc.abstractmethod
    def parse(self, tf_operation, tf_session):
        pass
    
    def type_name(self):    
        return 'convolution'
    
    def to_model_entry(self):
        return '%s<float, %s, %s, %s, %s>(weight%d)' % (self.type_name(), components(self.components_input, self.components_output), kernel(self.kernel_rows, self.kernel_columns), stride(self.stride_rows, self.stride_columns), padding(self.padding_rows, self.padding_columns), self.layer_index)
    
    def check_completeness(self):
        assert hasattr(self, 'weights') and self.weights is not None
        assert hasattr(self, 'components_input') and self.components_input is not None
        assert hasattr(self, 'components_output') and self.components_output is not None        
        assert hasattr(self, 'kernel_rows') and self.kernel_rows is not None
        assert hasattr(self, 'kernel_columns') and self.kernel_columns is not None        
        assert hasattr(self, 'stride_rows') and self.stride_rows is not None
        assert hasattr(self, 'stride_columns') and self.stride_columns is not None    
        assert hasattr(self, 'padding_rows') and self.padding_rows is not None    
        assert hasattr(self, 'padding_columns') and self.padding_columns is not None 
    
     
class PointwiseConvolution(WeightedLayer):
    
    @abc.abstractmethod
    def parse(self, tf_operation, tf_session):
        pass

    def type_name(self):    
        return 'pointwise_convolution'
    
    def to_model_entry(self):
        return '%s<float, %s>(weight%d)' % (self.type_name(), components(self.components_input, self.components_output), self.layer_index)
    
    def check_completeness(self):
        assert hasattr(self, 'weights') and self.weights is not None
        assert hasattr(self, 'components_input') and self.components_input is not None
        assert hasattr(self, 'components_output') and self.components_output is not None    
    
    
class DepthwiseConvolution(WeightedLayer):
        
    @abc.abstractmethod        
    def parse(self, tf_operation, tf_session):
        pass
    
    def type_name(self):    
        return 'depthwise_convolution'    

    def to_model_entry(self):
        return '%s<float, %s, %s, %s, %s>(weight%d)' % (self.type_name(), components(self.components_input), kernel(self.kernel_rows, self.kernel_columns), stride(self.stride_rows, self.stride_columns), padding(self.padding_rows, self.padding_columns), self.layer_index)        

    def check_completeness(self):
        assert hasattr(self, 'weights') and self.weights is not None
        assert hasattr(self, 'components_input') and self.components_input is not None
        assert hasattr(self, 'components_output') and self.components_output is not None        
        assert hasattr(self, 'kernel_rows') and self.kernel_rows is not None
        assert hasattr(self, 'kernel_columns') and self.kernel_columns is not None        
        assert hasattr(self, 'stride_rows') and self.stride_rows is not None
        assert hasattr(self, 'stride_columns') and self.stride_columns is not None    
        assert hasattr(self, 'padding_rows') and self.padding_rows is not None    
        assert hasattr(self, 'padding_columns') and self.padding_columns is not None 
    
    
class Bias(WeightedLayer):
        
    @abc.abstractmethod        
    def parse(self, tf_operation, tf_session):
        pass
        
    def type_name(self):    
        return 'bias'

    def to_model_entry(self):
        return '%s<float, %s>(weight%d)' % (self.type_name(), components(self.components), self.layer_index)
        
    def check_completeness(self):
        assert hasattr(self, 'weights') and self.weights is not None        
        assert hasattr(self, 'components') and self.components is not None

    
class BatchNorm(WeightedLayer):
        
    @abc.abstractmethod        
    def parse(self, tf_operation, tf_session):
        pass
    
    def type_name(self):    
        return 'batch_norm'

    def to_model_entry(self):
        return '%s<float, %s>(bnbeta%d, bnmean%d, bngamma_variance_epsilon%d)' % (self.type_name(), components(self.components), self.layer_index, self.layer_index, self.layer_index)  
        
    def check_completeness(self):
        assert hasattr(self, 'components') and self.components is not None
        assert hasattr(self, 'beta_weights') and self.beta_weights is not None  
        assert hasattr(self, 'moving_mean_weights') and self.moving_mean_weights is not None   
        assert hasattr(self, 'gamma_weights') and self.gamma_weights is not None   
        assert hasattr(self, 'moving_variance_weights') and self.moving_variance_weights is not None
        assert hasattr(self, 'epsilon') and self.epsilon is not None        
        
        
    def dump_weights(self, layer_index, weights_dir_path):

        # fuse gamma_weights, moving_variance_weights and epsilon
        self.gamma_variance_epsilon_weights = np.asarray([1/math.sqrt(val_variance + self.epsilon) * val_gamma for val_variance, val_gamma in zip(self.moving_variance_weights, self.gamma_weights)])
        
        self._WeightedLayer__dump_weights('bnbeta', layer_index, self.beta_weights, weights_dir_path, layer_var_name = 'bnbeta')
        self._WeightedLayer__dump_weights('bnmean', layer_index, self.moving_mean_weights, weights_dir_path, layer_var_name = 'bnmean')
        self._WeightedLayer__dump_weights('bngamma_variance_epsilon', layer_index, self.gamma_variance_epsilon_weights, weights_dir_path, layer_var_name = 'bngamma_variance_epsilon')
  

class BatchNormRelu(BatchNorm):
    def __init__(self, batch_norm_layer, relu_layer):
        self.components = batch_norm_layer.components
        self.beta_weights = batch_norm_layer.beta_weights
        self.moving_mean_weights = batch_norm_layer.moving_mean_weights
        self.gamma_weights = batch_norm_layer.gamma_weights
        self.moving_variance_weights = batch_norm_layer.moving_variance_weights
        self.epsilon = batch_norm_layer.epsilon
        
    def type_name(self):    
        return 'batch_norm_relu'
    
    def parse(self):
        pass
    

class BatchNormRelu6(BatchNormRelu):
    def __init__(self, batch_norm_layer, relu_layer):
        super(BatchNormRelu6, self).__init__(batch_norm_layer, relu_layer)

    def to_model_entry(self):
        return '%s<float, %s>(bnbeta%d, bnmean%d, bngamma_variance_epsilon%d, 0, 6)' % (super(BatchNormRelu6, self).type_name(), components(self.components), self.layer_index, self.layer_index, self.layer_index) 
    
    
class BiasRelu(Bias):
    def __init__(self, bias_layer, relu_layer):
        self.weights = bias_layer.weights        
        self.components = bias_layer.components

    def type_name(self):    
        return 'bias_relu'

    def parse(self):
        pass
    

class BiasRelu6(BiasRelu):
    def __init__(self, bias_layer, relu_layer):
        super(BiasRelu6, self).__init__(bias_layer, relu_layer)
    
    def to_model_entry(self):
        return '%s<float, %s>(weight%d, 0, 6)' % (super(BatchNormRelu6, self).type_name(), components(self.components), self.layer_index)    
        
    
class LayerFactory():
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod    
    def create_layer(self, op_type):
        pass