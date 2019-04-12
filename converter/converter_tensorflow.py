#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf
from converter.converter import *
from converter.layer_tensorflow import *

os.environ["CUDA_VISIBLE_DEVICES"]= '-1' # use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'  # INFO and WARNING messages are not printed


def convert(model_path, model_name, output_path = 'converted_models'):
    converter = ConverterTensorflow(model_path, model_name, output_path)
    converter.execute()

    
class ConverterTensorflow(Converter):
    def __init__(self, model_path, model_name, output_path):
        super(ConverterTensorflow, self).__init__(model_path, model_name, output_path)
        self.graph = None


    def load_model(self):        
        if not os.path.exists(self.model_path):
            raise ValueError('The specified file does not exist: {}'.format(self.model_path))

        graph_def = None
        try:
            with tf.gfile.GFile(self.model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        except BaseException as e:
            raise ValueError('Error loading the graph definition')

        try:
            assert graph_def is not None
            tf.import_graph_def(graph_def, name='')
            self.graph = tf.get_default_graph()
        except BaseException as e:
            raise e
        
            
    def get_tensors_with_weights(tf_operation):
        return [op_input for op_input in tf_operation.inputs if op_input.name.endswith('read:0')]    
            
            
    def parse_layers(self):
        layer_factory = self.layer_factory()
        layers = []

        constant_operations = ['Const', 'Placeholder','Identity', 'Shape', 'Squeeze', 'Reshape']
        
        with tf.Session() as sess:
            operations = [op for op in sess.graph.get_operations() if not op.type in constant_operations]

            for op in operations:
                layer = layer_factory.create_layer(op)
                if layer is not None:
                    layer.parse(op, sess)
                    layers.append(layer)
                else:
                    raise ValueError('Not supported layer: {}'.format(op))
        return layers
    
    
    def layer_factory(self):
        return LayerTensorflowFactory()
