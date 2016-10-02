# Papers
I implemented Charlie Tsai "Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks".
http://cs231n.stanford.edu/reports2016/262_Report.pdf

This is just a VGG-like convnet.
https://arxiv.org/abs/1409.1556

# Librarys
You need following librarys.
- Anaconda (Python 3.5)
- Theano 0.8.2 or TensorFlow 0.10.0
- Keras 1.1.0
- scikit-learn 0.17.1 (included in Anaconda)
- CUDA 7.5
- cuDNN 5.0

# Config files
### ~/.theanorc
```
[global]
floatX = float32
device = gpu

[lib]
cnmem = 1

[nvcc]
flags=-D_FORCE_INLINES
```

### ~/.keras/keras.json

Theano backend
```
{
    "epsilon": 1e-07,
    "image_dim_ordering": "th",
    "floatx": "float32",
    "backend": "theano"
}
```
or TensorFlow backend.
```
{
    "epsilon": 1e-07,
    "image_dim_ordering": "tf",
    "floatx": "float32",
    "backend": "tesorflow"
}
```
Modify both ```image_dim_ordering``` and ```backend```.
Theano 0.8.2 is faster than TensorFlow 0.10.0.

# Dataset
Please download dataset from http://etlcdb.db.aist.go.jp/?page_id=651 and extract to ETL8G folder.
Dataset contains 160 person hiragana characters.

# How to run
Just run ```python learn.py```.
