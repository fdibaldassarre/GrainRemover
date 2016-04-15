#!/usr/bin/env python3

# Simple Convolutional Neural Network class

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy

from PIL import Image
from PIL import ImageOps

import json

LEARNING_RATE = 0.1
MOMENTUM = 0.9

BATCH_SIZE = 30

import time

rng = numpy.random.RandomState(int(time.time()))

class NN():
  
  def __init__(self, layers_dims, learning_rate=None, momentum=None):
    self.learning_rate = LEARNING_RATE if learning_rate is None else learning_rate
    self.momentum = MOMENTUM if momentum is None else momentum
    self.layers_dims = layers_dims
    self.layers_number = len(self.layers_dims) - 1
    self.batch_size = BATCH_SIZE
    self.use_local_learning_rates = True
  
  def setLearningRate(self, rate):
    self.learning_rate = rate
  
  def useLocalLearningRates(self, val=True):
    self.use_local_learning_rates = val
  
  def setMomentum(self, momentum):
    self.momentum = momentum
  
  def setBatchSize(self, size):
    self.batch_size = size
  
  def setupInput(self):
    self.input = T.dmatrix(name='input')
  
  def setInput(self, var):
    self.input = var
    self.loadLayers()
  
  def getExpectedResult(self):
    return T.dmatrix(name='expected')
  
  def getLayersShapes(self):
    # set up model shapes
    layers_shapes = []
    for i in range(self.layers_number):
      input_size = self.layers_dims[i]
      output_size = self.layers_dims[i+1]
      layers_shapes.append({})
      layers_shapes[i]['W'] = (input_size, output_size)
      layers_shapes[i]['B'] = (output_size,)
    return layers_shapes
  
  def initializeModel(self):
    # Initialize the model
    # Setup layers sizes
    layers_shapes = self.getLayersShapes()
    # Input
    self.setupInput()
    # Weights and bias
    self.parameters = []
    for shapes in layers_shapes:
      if len(shapes) > 0:
        self.parameters.append(self.randomInitParameters(shapes))
      else:
        self.parameters.append(())
    # Layers
    self.loadLayers()
  
  def loadModel(self, model):
    self.setupInput()
    self.parameters = []
    for layer in model:
      if len(layer) > 0:
        weights = layer['W']
        bias = layer['B']
        w = theano.shared(numpy.asarray(weights, dtype=self.input.dtype))
        b = theano.shared(numpy.asarray(bias, dtype=self.input.dtype))
        self.parameters.append((w, b))
      else:
        self.parameters.append(())
    self.loadLayers()
  
  def loadLayers(self):
    self.layers = [self.input]
    for i in range(len(self.parameters)):
      prev = self.layers[-1]
      if len(self.parameters[i]) > 0:
        w, b = self.parameters[i]
        l = T.nnet.sigmoid(T.dot(prev, w) + b)
      else:
        # layer with no parameters (should never happer with NN,
        # can happen with CNN if max-pooling layer)
        l = self.createNoParametersLayer(i, prev)
      self.layers.append(l)
  
  def getLocalLearningRates(self):
    llr = [numpy.float32(self.learning_rate)] * self.layers_number
    if self.use_local_learning_rates:
      for layer in range(self.layers_number):
        input_size = self.layers_dims[layer]
        llr[layer] = numpy.float32( self.learning_rate / numpy.sqrt(input_size) )
    return llr
  
  def createNoParametersLayer(self, i, prev):
    return None
  
  def randomInitParameters(self, shapes):
    w_shape = shapes['W']
    b_shape = shapes['B']
    # Set weights bound
    a, b = w_shape
    w_bound = numpy.sqrt(a*b)
    # Initalize weights randomly with values in [-w_bound, w_bound]
    w = theano.shared(numpy.asarray(
                              rng.uniform(
                                low=-1.0 / w_bound,
                                high=1.0 / w_bound,
                                size=w_shape),
                              dtype=self.input.dtype))
    # Initialize bias to zero
    b = theano.shared(numpy.zeros(b_shape, dtype=self.input.dtype))
    return (w, b)
  
  ## Model LOAD/SAVE to file
  def loadModelFromFile(self, filepath):
    # Json decoder
    decoder = json.JSONDecoder()
    # Read the model from json file
    hand = open(filepath, 'r')
    tmp = []
    for line in hand:
      line = line.strip()
      tmp.append(line)
    hand.close()
    tmp = ''.join(tmp)
    # Decode
    model = decoder.decode(tmp)
    # Load the model
    self.loadModel(model)

  def saveModel(self, filepath):
    # Extract parameters
    model = []
    for params in self.parameters:
      if len(params) > 0:
        ws, bs = params
        w = ws.get_value().tolist()
        b = bs.get_value().tolist()
        # get sizes
        model.append({'W':w, 'B':b})
      else:
        model.append({})
    # Save to file
    encoder = json.JSONEncoder()
    data = encoder.encode(model)
    hand = open(filepath, 'w')
    hand.write(data)
    hand.close()
  
  ## Build functions
  def createCostFunction(self, prediction, expected):
    return T.sum((prediction-expected)**2) / self.batch_size
  
  def buildPredictFunction(self):
    prediction = self.layers[-1]
    return theano.function([self.input], prediction)
  
  def buildValidationFunction(self):
    # n is the number of images in the valildation set
    prediction = self.layers[-1]
    expected = self.getExpectedResult()
    cost_function = self.createCostFunction(prediction, expected)
    return theano.function([self.input, expected], cost_function)
  
  def buildTrainingFunction(self, cost_function=None):
    expected = self.getExpectedResult()
    
    # Create cost function
    if cost_function is None:
      prediction = self.layers[-1]
      cost_function = self.createCostFunction(prediction, expected)
    
    # initialize deltas
    deltas = []
    layer = 0
    for lparams in self.parameters:
      for param in lparams:
        shp = param.get_value().shape
        delta = theano.shared(numpy.zeros(shp, dtype=self.input.dtype))
        deltas.append((layer, param, delta))
      layer += 1
    # compute local learning rates
    llr = self.getLocalLearningRates()

    updates = []
    for delta_d in deltas:
      nlayer, param, delta = delta_d
      new_delta = self.momentum * delta - llr[nlayer] * T.grad(cost_function, param)
      updates.append( (param, param + new_delta ) )
      updates.append( (delta, new_delta) )
    
    return theano.function([self.input, expected], cost_function, updates=updates)
  
  ## Conversion Image->Data
  def convertImageToData(self, img):
    # Convert to numpy array/matrix
    img_n = numpy.asarray(img, dtype=numpy.float32) / 255.
    # Reshape
    w, h = img.size
    img_r = img_n.reshape((w*h))
    return img_r
  
  ## Conversion Data->Image
  def convertDataToImage(self, img_data, img_shape):
    # Rescale to [0, 255]
    img_data = numpy.minimum(numpy.maximum(0., img_data), 1.)
    img_data = numpy.uint8(img_data * 255.)
    img_r = img_data.reshape(img_shape)
    return Image.fromarray(img_r)

  
class CNN(NN):
  
  def __init__(self, layers_dims, layers_conv_sizes, learning_rate=None, momentum=None):
    NN.__init__(self, layers_dims, learning_rate, momentum)
    self.layers_conv_sizes = layers_conv_sizes
    self.input_channels = self.layers_dims[0]
    self.image_padding = 0
    self.downscale_factor = 0
    for f in self.layers_conv_sizes:
      if f > 0:
        self.image_padding += (f-1) * (0.5 ** (1-self.downscale_factor))
      else:
        self.downscale_factor += 1
    assert numpy.floor(self.image_padding) == numpy.ceil(self.image_padding)
    self.image_padding = int(self.image_padding)
  
  def setupInput(self):
    self.input = T.tensor4(name='input')
  
  def getExpectedResult(self):
    return T.tensor4(name='expected')
  
  def getLocalLearningRates(self):
    llr = [numpy.float32(self.learning_rate)] * self.layers_number
    if self.use_local_learning_rates:
      for layer in range(self.layers_number):
        input_size = self.layers_dims[layer] * (self.layers_conv_sizes[layer] ** 2)
        llr[layer] = numpy.float32( self.learning_rate / numpy.sqrt(input_size) )
    return llr
  
  def getLayersShapes(self):
    # set up model shapes
    layers_shapes = []
    for i in range(self.layers_number):
      input_size = self.layers_dims[i]
      output_size = self.layers_dims[i+1]
      f = self.layers_conv_sizes[i]
      layers_shapes.append({})
      if f > 0:
        # convolution layer
        layers_shapes[i]['W'] = (output_size, input_size, f, f)
        layers_shapes[i]['B'] = (output_size,)
    return layers_shapes
  
  def loadLayers(self):
    self.layers = [self.input]
    for i in range(len(self.parameters)):
      prev = self.layers[-1]
      if len(self.parameters[i]) > 0:
        w, b = self.parameters[i]
        if i == self.layers_number - 1:
          l = conv.conv2d(prev, w) + b.dimshuffle('x', 0, 'x', 'x')
        else:
          l = T.maximum(0, conv.conv2d(prev, w) + b.dimshuffle('x', 0, 'x', 'x'))
      else:
        l = self.createNoParametersLayer(i, prev)
      self.layers.append(l)
  
  def createNoParametersLayer(self, i, prev):
    return self.createMaxpoolingLayer(i, prev)
  
  def createMaxpoolingLayer(self, i, prev):
    maxpool_shape = (2, 2)
    output = T.signal.downsample.max_pool_2d(prev, maxpool_shape, ignore_border=False)
    return output
  
  def randomInitParameters(self, shapes):
    w_shape = shapes['W']
    b_shape = shapes['B']
    # Set weights bound
    _, i, a, b = w_shape
    w_bound = numpy.sqrt(i*a*b)
    # Initalize weights randomly with values in [-1/w_bound, 1/w_bound]
    w = theano.shared(numpy.asarray(
                              rng.uniform(
                                low=-1.0 / w_bound,
                                high=1.0 / w_bound,
                                size=w_shape),
                              dtype=self.input.dtype))
    # Initialize bias to zero
    b = theano.shared(numpy.zeros(b_shape, dtype=self.input.dtype))
    return (w, b)
  
  ## Conversion Image->Data
  def convertImageToData(self, img, padding=False):
    # Convert to RGB or B/W
    if self.input_channels == 1:
      img = img.convert('L')
    else:
      img = img.convert('RGB')
    # Add padding
    if padding:
      img = ImageOps.expand(img, border=self.image_padding, fill='black')
    w, h = img.size
    img_n = numpy.asarray(img, dtype=numpy.float32) / 255.
    # Transponse if RGB
    if self.input_channels == 3:
      # img shape = h, w, channels
      # tensor shape = channels, h, w
      img_n = img_n.transpose(2, 0, 1)
    # Reshape
    img_n = img_n.reshape(1, self.input_channels, h, w)
    return img_n
  
  ## Conversion Data->Image
  def convertDataToImage(self, img_data):
    # Conversion tensor --> PIL.Image
    img = img_data[0, :, :, :]
    # Transpose
    if self.input_channels == 3:
      # tensor shape = channels, h, w
      # image shape = h, w, channels
      img = img.transpose(1, 2, 0)
    else:
      img = img[0, :, :]
    # Rescale to [0, 255]
    img = numpy.minimum(numpy.maximum(0., img), 1.)
    img = numpy.uint8(numpy.round(img * 255.))
    return Image.fromarray(img)
  
  ## Print function
  # TODO: better or delete
  def testConsistency(self):
    print('# Layers dimensions')
    print(len(self.layers_dims) == len(self.layers_conv_sizes) + 1)
    print(len(self.layers_dims) == len(self.layers))
    print(len(self.parameters) == len(self.layers_dims) - 1)
    print('# Data consistency')
    for layer in range(len(self.parameters)):
      params = self.parameters[layer]
      print('# Layer:', layer)
      if len(params) > 0:
        w, b = params
        o, i, f, g = w.get_value().shape
        ob = b.get_value().shape[0]
        print(f==g)
        print(ob==o)
        fe = self.layers_conv_sizes[layer]
        print(fe==f)
        ie = self.layers_dims[layer]
        oe = self.layers_dims[layer+1]
        print(i==ie)
        print(o==oe)
    print('Done!')
  
  def printInfo(self):
    print('Layers:')
    for n in range(len(self.layers_dims)-1):
      input_size = self.layers_dims[n]
      conv_size = self.layers_conv_sizes[n]
      output_size = self.layers_dims[n+1]
      print(' ', input_size, '--', conv_size, '-->', output_size)
