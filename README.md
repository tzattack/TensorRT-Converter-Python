# TensorRT-Converter-Python
Convert Caffe/PyTorch/TensorFlow model to TensorRT with fast int8 and fp16 in Python3.


## install
use `sh install.sh` to install requirements.  
download TensorRT lib for Nvidia official site and place it in a dir where is convenient.  
use `export LD_LIBRARY_PATH:/path/to/trt/lib:$LD_LIBRARY_PATH` to tell system where is the TensorRT lib.  

## usage
place your model in `input_model` dir and remove unnecessary models in this dir.  
place your calibration data in `calibration_data` and remove unnecessary calibration data in this dir.  
place your test data in `test_data` dir.  
  
use `--onnx` to convert a pytorch onnx model.  
use `--caffe` to convert a caffe model.  
use `--pb` to convert a tensorflow frozen graph model.

