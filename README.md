# GrainRemover
Convolutional Neural Network for grain removal from digital images (Work in progess).

Installation requirements
------------

- Python
- Python Imaging Library (PIL)
- Theano (http://www.deeplearning.net/software/theano/install.html)

Usage
------------

To clean a noisy image run:

```sh
./predict.py --input noisy_image.png --output new_image.png
```

Note: the model used atm is not fully trained and works only with grayscale images.

Note2: you can obtain a similar result using a selective gaussian blur.

Running on the GPU
------------

To run the program on the GPU (much faster) use:

```sh
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
```

see http://www.deeplearning.net/software/theano/install.html#using-the-gpu for more info.

Examples
------------

Noisy image:

![input](https://raw.githubusercontent.com/fdibaldassarre/GrainRemover/master/examples/noisy.png)

Result:

![input](https://raw.githubusercontent.com/fdibaldassarre/GrainRemover/master/examples/result.png)

The noisy image was created by using the GIMP filter "Add Film Grain" on a clean SVG image.

Training
------------

## Training data

Put the noisy images in data/images/noise/ and the clean ones in data/images/clean/.

To each image in the clean folder must correspond another image with the same name and size in the
noise folder.

Image with name starting with # are ignored.

Use PNG images.

## NN setup

To change the network paramenters (number of layers, convolution size, ...) edit cnn/GrainExtractor.py

To tweak the training parameters (learning rate, momentum, training epochs, ...) edit train.py

## Training

Run:
```sh
./train.py
```

The first layers are pretrained as a stacked denoising auto encoder and then the network is finetuned.

The backup files are in models/backup and the model with the best validation score is saved in models/model_train.json

## Test

To test the model put some noisy images in data/images/test/noise and the corresponding clean ones
in data/images/test/clean and then run:

```sh
./test.py
```
