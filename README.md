# SJTU-M3DV

Final project for course EE369 in SJTU, implemented in Keras+Tensorflow.

The final submission results are in folder BestResult, including Weight file of the trained model "weights.106.h5" and the forecast file "submit.csv".

## Requirements

* Python 3 (Anaconda2/3.5.0 specifically)
* Tensorflow-gpu == 2.0.0
* Keras == 2.3.1

## Code Structure
[DenseNet.py](https://github.com/sibyl592/SJTU-M3DV/blob/master/DenseNet.py) : Training network, implementation of the paper Densely Connected Convolutional Networks in Keras

[data.py](https://github.com/sibyl592/SJTU-M3DV/blob/master/data.py) : Data processing

[train.py](https://github.com/sibyl592/SJTU-M3DV/blob/master/train.py) : Parameter configuration and training

[DenseSharp-master](https://github.com/sibyl592/SJTU-M3DV/tree/master/DenseSharp-master/DenseSharp-master) : A parameter-efficient 3D DenseNet-based deep neural network, with multi-task learning the nodule classification labels and segmentation masks.

[test.py](https://github.com/sibyl592/SJTU-M3DV/blob/master/test.py) : Run this file to directly output a submission.csv file for kaggle submission
