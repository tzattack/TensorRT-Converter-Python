# TensorRT-Converter-Python
Convert Caffe/PyTorch/TensorFlow model to TensorRT 7.2 with fast int8 and fp16 in Python3.


## Install
Use `sh install.sh` to install requirements.  

## Usage
Place your model in `input_model` dir and remove unnecessary models in this dir.  
Place your calibration data in `calibration_data` and remove unnecessary calibration data in this dir.  
Place your test data in `test_data` dir.  
  
Use `--onnx` to convert a pytorch onnx model.  
Use `--caffe` to convert a caffe model.   
Use `--pb` to convert a tensorflow frozen graph model.  

Use `--nchw` to mark input dimension.  
Use `--dynamic` to convert a dynamic model.  
Use `--cache` if you want to reuse calibration cache.  

Use `--fp16` to convert model in fast fp16 mode.  
Use `--int8` to convert model in fast int8 mode.  

Modify `preprocess()` in `trtconvert.py` to fit your model's preprocessing.  

## Example
Use `python3 trtconvert.py --onnx --nchw 32,3,300,300 --dynamic --int8` to convert an onnx model with dynamic input in 32 batch size to int8 trt model.


## Known issues
- Accuracy descends when converting a model with depthwise convolution on Turing Graphic cards.
- Low chances for converting a tf model.
