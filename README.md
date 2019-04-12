# PolimiDL Converter

__PolimiDL Converter__ is a Deep Learning model converter to support the use of __PolimiDL__ inference framework (https://github.com/darianfrajberg/polimidl).

It takes in input a model trained with the Deep Learning framework of your choice and generates as output a model with __PolimiDL__'s compatible format for its deployment.


## Supported frameworks
The current supported __Deep Learning frameworks__ in __PolimiDL Converter__ are:
* TensorFlow


## Installation
To install the required modules:
```
# Install tensorflow
pip install tensorflow

# Install abc for abstract methods definition
pip install abcplus
```


## Running the converter
To convert a model trained on TensorFlow:
```
python main.py --model_path MODEL_PATH --model_name MODEL_NAME --output_path OUTPUT_PATH
```


### MobileNet conversion example
To convert MobileNet model trained on TensorFlow:
```
python main.py --model_path models/mobilenet_v1_1.0_224_frozen.pb --model_name mobilenet
```


## Extending supported frameworks
__PolimiDL Converter__ presents a generic and easily extensible architecture.
To support the conversion of models trained with additional frameworks, you should extend the abstract classes and corresponding abstract methods defined in the following modules:
* `converter/converter.py`
* `converter/layer.py`

You can look at the implementation for TensorFlow support:
* `converter/converter_tensorflow.py`
* `converter/layer_tensorflow.py`


## Extending supported layers
The extension of supported layers is coupled to the support of such layers in __PolimiDL__.
If a new layer is introduced into __PolimiDL__, then you can support such layer by defining its generic implementation in `converter/layer.py` and its custom framework implementation in the corresponding module extension (e.g. `converter/layer_tensorflow.py`).

