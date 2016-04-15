#!/usr/bin/env python3

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy

from PIL import Image
from PIL import ImageOps

from cnn.AutoEncoders import CSdA
from cnn.Main import rng

CHANNELS = 1
LAYERS_DIMS = [CHANNELS, 32, 32, CHANNELS]
LAYERS_CONVS = [5, 5, 3]

LEARNING_RATE = 0.001
MOMENTUM = 0.9

CORRUPTION_PER_LAYER = 0.1

IMAGE_SIZE = 200

class GrainRemover(CSdA):
  
  def __init__(self):
    CSdA.__init__(self, LAYERS_DIMS, LAYERS_CONVS, CORRUPTION_PER_LAYER, learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    self.image_size = IMAGE_SIZE
    factor = 0.5 ** self.downscale_factor
    self.input_image_size = int(self.image_size + 2*self.image_padding)
    self.output_image_size = int(self.image_size * factor)
  
  ## Input setup pretraing:
  def setupInputPretraining(self, m, data):
    # Remove excess padding
    padding = 0
    for layer in range(m):
      padding += (self.layers_conv_sizes[layer] - 1) / 2
    # Last layer has conv size as the preceding one
    padding += (self.layers_conv_sizes[m-1] - 1) / 2
    diff = self.image_padding - padding
    if diff > 0:
      return data[:, :, diff:-diff, diff:-diff]
    else:
      return data
  
  def clearPadding(self, data):
    return data[:, :, self.image_padding:-self.image_padding, self.image_padding:-self.image_padding]
    
  ## Create Training data
  def splitInputImageModel(self, imgpath):
    return self.splitImageModel(imgpath, is_input=True)
    
  def splitOutputImageModel(self, imgpath):
    return self.splitImageModel(imgpath, is_input=False)
  
  def splitImageModel(self, imgpath, is_input):
    models = []
    # Load image
    img = Image.open(imgpath)
    # Convert to RGB or B/W
    if self.input_channels == 1:
      img = img.convert('L')
    else:
      img = img.convert('RGB')
    # Split in multiple parts
    w, h = img.size
    n = int(w / self.image_size)
    m = int(h / self.image_size)
    for i in range(n):
      for j in range(m):
        x = i*self.image_size
        y = j*self.image_size
        simg = img.crop((x, y, x+self.image_size, y+self.image_size))
        if not is_input and self.downscale_factor > 0:
          # downscale the image by (1/2)^self.downscale_factor
          simg = simg.resize((self.output_image_size, self.output_image_size), Image.ANTIALIAS)
        img_model = self.convertImageToData(simg, is_input)[0, :, :, :]
        models.append(img_model)
    return models
  
  def createTrainingData(self, input_images, output_images):
    assert len(input_images) == len(output_images)
    n = len(input_images)
    # Split images in pieces of size self.image_size * self.image_size
    input_models = []
    output_models = []
    for i in range(n):
      input_models.extend(self.splitInputImageModel(input_images[i]))
      output_models.extend(self.splitOutputImageModel(output_images[i]))
    # Construct training and expectation tensor
    m = len(input_models)
    training = numpy.ndarray((m, self.input_channels, self.input_image_size, self.input_image_size), dtype=numpy.float32)
    expectation = numpy.ndarray((m, self.input_channels, self.output_image_size, self.output_image_size), dtype=numpy.float32)
    for i in range(m):
      training[i] = input_models[i]
      expectation[i] = output_models[i]
    return training, expectation, m
  
  def shuffleTrainingData(self, training, expectation, m, new_indexes):
    new_training = numpy.ndarray((m, self.input_channels, self.input_image_size, self.input_image_size), dtype=numpy.float32)
    new_expectation = numpy.ndarray((m, self.input_channels, self.output_image_size, self.output_image_size), dtype=numpy.float32)
    for i in range(m):
      j = new_indexes[i]
      new_training[i] = training[j]
      new_expectation[i] = expectation[j]
    return new_training, new_expectation
  
  def convertInputImageToData(self, path):
    img = Image.open(path)
    return self.convertImageToData(img, padding=True)
