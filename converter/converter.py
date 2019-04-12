#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import shutil
import abc
from converter.layer import *
import converter.utils as utils
from collections import OrderedDict

# Dictionary to map the class to use for the corresponding fusion of layers
layers_fusion_dict = OrderedDict([
    (BatchNormRelu, [BatchNorm, Relu]),
    (BatchNormRelu6, [BatchNorm, Relu6]), 
    (BiasRelu, [Bias, Relu]),
    (BiasRelu6, [Bias, Relu6])
])


class Converter(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_path, model_name, output_path):
        self.model_path = model_path
        self.model_name = model_name
        self.output_path = output_path
        
        self.output_dir_path = output_path + '/' + model_name
        self.weights_dir_path = self.output_dir_path + '/' + model_name + '_weights'
        
        

    def execute(self):
        print('Loading model...')
        self.load_model()
        print('Parsing layers...')
        self.layers = self.parse_layers()
        print('Checking model completeness...')
        self.__check_model_completeness()
        print('Fusing layers...')
        self.__fuse_layers()        
        print('Extracting weights...')
        self.__generate_output_folder()        
        self.__extract_weights()
        print('Generating model structure...')
        self.__dump_model_structure()
        print('Conversion finished.')        
     
        
    @abc.abstractmethod    
    def load_model(self):
        pass
    

    @abc.abstractmethod    
    def parse_layers(self):
        pass
    
    @abc.abstractmethod    
    def layer_factory(self):
        pass
    
    def __generate_output_folder(self):
        shutil.rmtree(self.output_dir_path, ignore_errors=True)
        os.makedirs(self.weights_dir_path)


    def __extract_weights(self):
        layer_index = 0        
        for layer in self.layers:
            if layer.has_weights():
                layer_index += 1
                layer.dump_weights(layer_index, self.weights_dir_path)                


    def __check_model_completeness(self):
        for layer in self.layers:
            layer.check_completeness()


    def __fuse_layers(self):
        normalized_layers = [layer.__class__.__bases__[0] for layer in self.layers]
        
        for fused_layer, layers_pattern in layers_fusion_dict.items():
            indexes = utils.find_occurrences(normalized_layers, layers_pattern)
            for index in indexes:
                # replace first layer in the pattern occurrence with fused layer
                layers_to_fuse = self.layers[index:index+len(layers_pattern)]
                self.layers[index] = fused_layer(*layers_to_fuse)
                # set consecutive layers to None
                for i in range(1, len(layers_to_fuse)):
                    normalized_layers[index+i] = None
                    self.layers[index+i] = None                    
        # filter out layers set to None
        self.layers = [layer for layer in self.layers if layer is not None]

        
    def __dump_model_structure(self):
        file_path = self.output_dir_path + '/'+ self.model_name +'.hpp'
        layer_types = set([layer.type_name() for layer in self.layers])        
        
        f = open(file_path, 'w')
        f.write('#include <polimidl/network.hpp>\n')
        for layer_type in layer_types:
            f.write('#include <polimidl/layers/%s.hpp>\n' % (layer_type))
        f.write('\n')
            
        for layer_file in utils.natural_sort(glob.glob(self.weights_dir_path+"/*")):
            f.write('#include \"./%s_weights/%s"\n' % (self.model_name, os.path.basename(layer_file)))
        f.write('\n')
        
        f.write('auto build_%s(std::size_t h, std::size_t w, unsigned int number_of_workers) ' % (self.model_name) + '{\n')
        f.write('\tusing polimidl::build_network;\n')
        f.write('\tusing polimidl::layers::components;\n')
        f.write('\tusing polimidl::layers::kernel;\n')
        f.write('\tusing polimidl::layers::padding;\n')
        f.write('\tusing polimidl::layers::stride;\n')
        for layer_type in layer_types:
            f.write('\tusing polimidl::layers::%s;\n' % (layer_type))
        f.write('\n')
        
        f.write('\treturn build_network<float>(h, w, number_of_workers,\n')
        layers_formatted = ',\n'.join(['\t\t%s' % (layer.to_model_entry()) for layer in self.layers])
        f.write(layers_formatted)
        f.write('\n\t);\n')
        f.write('}\n')        
        f.close()
