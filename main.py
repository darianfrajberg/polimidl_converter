#!/usr/bin/python

import argparse

from converter.converter_tensorflow import convert


def main(model_path, model_name, output_path):
    convert(model_path, model_name, output_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Script to run PolimiDL converter')
    parser.add_argument('--model_path', help='Insert model path of TensorFlow .pb frozen graph')
    parser.add_argument('--model_name', help='Insert model name')
    parser.add_argument('--output_path', help='Insert output path', default = 'converted_models') 
    args = parser.parse_args()

    main(args.model_path, args.model_name, args.output_path)