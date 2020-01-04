# SJTU-M3DV

Final project for course EE369 in SJTU, implemented in Keras+Tensorflow.

The final submission results are in folder BestResult, including Weight file of the trained model "weights.106.h5" and the forecast file "submit.csv".

## Requirements

* Python 3 (Anaconda2/3.5.0 specifically)
* Tensorflow-gpu == 2.0.0
* Keras == 2.3.1

## Code Structure
DenseNet.py : training network, implementation of the paper Densely Connected Convolutional Networks in Keras

data.py : data processing

train.py : parameter configuration and training

DenseSharp-master : a parameter-efficient 3D DenseNet-based deep neural network, with multi-task learning the nodule classification labels and segmentation masks.
