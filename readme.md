# What is this?
Japanese hand written Hiragana (ひらがな) recoginition by deep learning. Accuracy is **99.78%**.

# Papers
My implementation is based on Charlie Tsai's "Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks".
http://cs231n.stanford.edu/reports2016/262_Report.pdf

This is just a VGG-like convnet.
https://arxiv.org/abs/1409.1556

# What I modified
- Resized image to 32x32 px. Charlie Tsai's paper is 64x64 px.
- Initial weights are from a normal distribution with 0.1 standard deviation.
- Used AdaDelta and increased the number of epochs to 400.
- Add random rotation and zoom to images.

By these changes the accuracy increased from 96.13% to 99.78%.

# Librarys
You need following librarys.
- Anaconda (Python 3.5)
- Theano 0.8.2 or TensorFlow 0.10.0
- Keras 1.1.0
- scikit-learn 0.18.0 (included in Anaconda)
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
    "backend": "tensorflow"
}
```
Modify both ```image_dim_ordering``` and ```backend```.
Theano 0.8.2 is faster than TensorFlow 0.10.0.

# Dataset
Please download dataset from http://etlcdb.db.aist.go.jp/?page_id=651 and extract to ETL8G folder.
Dataset contains 160 person hiragana characters.

# How to run
Just run ```python learn.py```.
