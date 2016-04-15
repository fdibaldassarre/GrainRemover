#!/usr/bin/env python3

# Convolutional Denoising Autoencoder
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

import numpy

import json

from cnn.Main import CNN
from cnn.Main import rng

theano_rng = RandomStreams(rng.randint(2 ** 30))

class CdA(CNN):
  
  def __init__(self, layers_dims, conv_size, corruption_level=0.1, learning_rate=None, momentum=None):
    assert len(layers_dims) == 3
    CNN.__init__(self, layers_dims, [conv_size, conv_size], learning_rate, momentum)
    self.corruption_level = corruption_level
    self.use_local_learning_rates = False
    self.training_input = None
  
  '''
  def getEncodingFunction(self):
    base = T.tensor4(name='base')
    w, b = self.parameters[0]
    encoding = T.maximum(0, conv.conv2d(base, w) + b.dimshuffle('x', 0, 'x', 'x'))
    return theano.function([base], encoding)
  
  def getDecodingFunction(self, var):
    encoded = T.tensor4(name='encoded')
    w, b = self.parameters[1]
    decoding =  T.maximum(0, conv.conv2d(encoded, w) + b.dimshuffle('x', 0, 'x', 'x'))
    return theano.function([encoded], decoding)
  '''
  
  def setTrainingInput(self, var):
    self.training_input = var
  
  def setCorruptionLevel(self, lvl):
    self.corruption_level = lvl
  
  def corruptVariable(self, var):
    return theano_rng.binomial(size=var.shape, n=1, p=1-self.corruption_level, dtype=theano.config.floatX) * var
  
  def loadLayers(self):
    self.layers = [self.input]
    for i in range(2):
      prev = self.layers[-1]
      w, b = self.parameters[i]
      if i == 0:
        # corrupt input
        prev = self.corruptVariable(prev)
      l = T.maximum(0, conv.conv2d(prev, w) + b.dimshuffle('x', 0, 'x', 'x'))
      self.layers.append(l)
  
  def getExpectedResult(self):  
    return self.input
  
  def createCostFunction(self, prediction, expected):
    # Resize tensor to the same size as the result
    input_convolution = self.createResizeConvolution()
    clean = conv.conv2d(expected, input_convolution)
    return T.sum((prediction-clean)**2) / (self.batch_size * self.input_channels)
  
  def createResizeConvolution(self):
    # Create an identity convolution of 2*conv_size
    f = self.layers_conv_sizes[0]
    f = 2*(f-1) + 1
    i = self.layers_dims[0]
    identity = numpy.zeros((i, i, f, f), dtype=self.input.dtype)
    r = (f-1)/2
    identity[:, :, r, r] = 1.0
    input_convolution = theano.shared(identity)
    return input_convolution
  
  def buildTrainingFunction(self):
    # Create cost function
    prediction = self.layers[-1]
    cost_function = self.createCostFunction(prediction, self.input)
    
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
    
    if self.training_input is None:
      self.training_input = self.input
    
    return theano.function([self.training_input], cost_function, updates=updates)
  

class CSdA(CNN):
  
  def __init__(self, layers_dims, layers_conv_sizes, corruption_per_level, learning_rate=None, momentum=None):
    CNN.__init__(self, layers_dims, layers_conv_sizes, learning_rate, momentum)
    self.corruption_per_level = corruption_per_level
  
  def initializeModel(self):
    self.setupInput()
    self.initializeCdA()
    self.importParametersFromCda()
    self.loadLayers()
  
  def loadModels(self, models):
    self.setupInput()
    self.loadCdaFromModels(models)
    self.importParametersFromCda()
    self.loadLayers()
  
  def importParametersFromCda(self):
    self.parameters = []
    for da in self.cda:
      self.parameters.append(da.parameters[0])
  
  def initializeCdA(self):
    self.cda = []
    for layer in range(1, len(self.layers_dims)):
      x = self.layers_dims[layer-1]
      y = self.layers_dims[layer]
      layers_dims = (x, y, x)
      conv_size = self.layers_conv_sizes[layer-1]
      # create
      da = CdA(layers_dims, conv_size)
      da.initializeModel()
      # append
      self.cda.append(da)
  
  def loadCdaFromModels(self, models):
    self.cda = []
    for layer in range(1, len(models)+1):
      model = models[layer-1]
      # set base data
      x = self.layers_dims[layer-1]
      y = self.layers_dims[layer]
      layers_dims = (x, y, x)
      conv_size = self.layers_conv_sizes[layer-1]
      # create
      da = CdA(layers_dims, conv_size)
      da.loadModel(model)
      # append
      self.cda.append(da)
  
  # Save/Load function
  def saveModel(self, filepath):
    # save all the da parameters
    models = []
    for da in self.cda:
      model = []
      for params in da.parameters:
        ws, bs = params
        w = ws.get_value().tolist()
        b = bs.get_value().tolist()
        # get sizes
        model.append({'W':w, 'B':b})
      models.append(model)
    # Save to file
    encoder = json.JSONEncoder()
    data = encoder.encode(models)
    hand = open(filepath, 'w')
    hand.write(data)
    hand.close()
  
  def saveCNNModel(self, filepath):
    CNN.saveModel(self, filepath)
  
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
    models = decoder.decode(tmp)
    # Load the model
    self.loadModels(models)
  
  def loadCNNModelFromFile(self, filepath):
    CNN.loadModelFromFile(self, filepath)
  
  # Training functions  
  def buildLayerTrainingFunction(self, layer):
    # layer is in [1, len(self.layers_dims)-1]
    # NOTE: training of the last layer is useless
    ncda = layer - 1
    # Train self.cda[ncda]
    da = self.cda[ncda]
    da.setLearningRate(self.learning_rate)
    da.setMomentum(self.momentum)
    # Set corruption level
    corruption_level = layer * self.corruption_per_level
    da.setCorruptionLevel(corruption_level)
    # Set input
    base = self.layers[layer-1]
    da.setTrainingInput(self.input)
    da.setInput(base)
    # Compute Training function
    f = da.buildTrainingFunction()
    return f 
    
