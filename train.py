#!/usr/bin/env python3

import os
import sys
import json

path = os.path.abspath(__file__)
MAIN_FOLDER = os.path.dirname(path)
MODELS_FOLDER = os.path.join(MAIN_FOLDER, 'models/')
BACKUP_FOLDER = os.path.join(MODELS_FOLDER, 'backup/')
if not os.path.isdir(BACKUP_FOLDER):
  os.mkdir(BACKUP_FOLDER)

from cnn.GrainRemover import GrainRemover

import numpy

# Use rng to shuffle training data
# I want it deterministic because I want to
# keep training and validation data separated
# in all the training sessions 
rng = numpy.random.RandomState(1)

BACKUP_STEP = 2
VALIDATION_STEP = 5
TRAINING_EPOCHS = 100

# Pre-Training
PRETRAINING_EPOCHS = [20, 20, 20, 15]
PRETRAINING_LEARNING_RATES = [0.000001, 0.000001, 0.000001, 0.000001]
LAST_PRETRAINING_LAYER = 4

MINI_BATCH_SIZE = 15

def loadTrainData(filepath):
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
  return decoder.decode(tmp)

def saveTrainData(data, filepath):
  encoder = json.JSONEncoder()
  data_str = encoder.encode(data)
  hand = open(filepath, 'w')
  hand.write(data_str)
  hand.close()


def getValidationError(validation_f, validation_in, validation_out, batch_size):
  n_validation = len(validation_in)
  n_batches = int(numpy.ceil(n_validation / batch_size))
  error = 0
  for n in range(n_batches):
    start = n * batch_size
    end = (n+1) * batch_size
    # Note: The last batch has size <= batch_size
    if end >= n_validation:
      factor = batch_size / (end-start)
    else:
      factor = 1.0
    error += validation_f(validation_in[start:end], validation_out[start:end]) * factor
  error = error / n_batches
  return float(error)
  

cnn = GrainRemover()
cnn.setBatchSize(MINI_BATCH_SIZE)

# Set backup and model save files
model_backup_savepath = os.path.join(BACKUP_FOLDER, 'model_train.json')
model_savepath = os.path.join(MODELS_FOLDER, 'model_train.json')
if not os.path.exists(model_backup_savepath):
  print('Initialize new model')
  cnn.initializeModel()
else:
  print('Loading backup model')
  cnn.loadModelFromFile(model_backup_savepath)
  
# Load train data
train_status_savepath = os.path.join(BACKUP_FOLDER, 'train_status.json')
if os.path.exists(train_status_savepath):
  train_data = loadTrainData(train_status_savepath)
  print('Resume training:')
  print(' Last pretrained layer:', train_data['pretraining_layer'])
  print(' Best validation score:', train_data['best_validation'])
  print(' Trained for', train_data['last_train_epoch'], 'epochs')
else:
  train_data = {}
  train_data['best_validation'] = None
  train_data['last_train_epoch'] = 0
  train_data['pretraining_layer'] = 0

# Load training data
print('Loading samples...')
clean_folder = os.path.join(MAIN_FOLDER, 'data/images/clean')
noise_folder = os.path.join(MAIN_FOLDER, 'data/images/noise')

input_images = []
output_images = []
for name in os.listdir(clean_folder):
  if name.startswith('#'):
    continue
  cimage = os.path.join(clean_folder, name)
  nimage = os.path.join(noise_folder, name)
  assert os.path.exists(nimage)
  input_images.append(nimage)
  output_images.append(cimage)

# Create data
data_input, data_output, n = cnn.createTrainingData(input_images, output_images)
print(' - Got', n, 'samples')
# Shuffle data (deterministic!)
shuffled_indexes = list(range(n))
rng.shuffle(shuffled_indexes)
data_input, data_output = cnn.shuffleTrainingData(data_input, data_output, n, shuffled_indexes)

# Set training/validation data
n_train = int(n*0.80)
n_validation = n - n_train

training_in = data_input[:n_train, :, :, :]
training_out = data_output[:n_train, :, :, :]

validation_in = data_input[n_train:, :, :, :]
validation_out = data_output[n_train:, :, :, :]

max_batch_index = int(numpy.ceil(n_train / MINI_BATCH_SIZE))
print(' - There are', max_batch_index, 'mini-batches')

if max_batch_index > 5:
  error_print_step = int(max_batch_index / 5)
else:
  error_print_step = 1

# Pre-training as a stacked denoise autoencoder
print('Pre-training')
# Set momentum
cnn.setMomentum(0.9)

for layer in range(train_data['pretraining_layer']+1, LAST_PRETRAINING_LAYER+1):
  print('### Layer', layer, 'out of', LAST_PRETRAINING_LAYER)
  # Set learning rate NOTE: autoencoders do not use local learning rates
  learning_rate = PRETRAINING_LEARNING_RATES[layer-1]
  cnn.setLearningRate(learning_rate)
  # Build models
  print('Building models...')
  pretrain_f = cnn.buildLayerTrainingFunction(layer)
  pretrain_input = cnn.setupInputPretraining(layer, training_in)
  pretraining_epochs = PRETRAINING_EPOCHS[layer-1]
  for epoch in range(1, pretraining_epochs + 1):
    print('* Epoch:', epoch)
    for index in range(max_batch_index):
      b_start = index * MINI_BATCH_SIZE
      b_end = (index+1) * MINI_BATCH_SIZE
      error_train = pretrain_f(pretrain_input[b_start : b_end])
      if index % error_print_step == 0:
        print('  -', error_train)
    ## DEBUG -- start
    if epoch % 5 == 0:
      # backup the model
      print('--- Backup model ---')
      cnn.saveModel(model_backup_savepath)
    ## DEBUG -- end
  print('Pretraining complete for layer', layer, ' !')
  # backup the model
  print('--- Backup model ---')
  cnn.saveModel(model_backup_savepath)
  # save training data
  train_data['pretraining_layer'] = layer
  saveTrainData(train_data, train_status_savepath)
print('Pre-training complete!')

print('Fine-tuning')
# Set learning rate and momentum
cnn.setLearningRate(0.000001)
cnn.setMomentum(0.9)

# Build training and validation functions
print('Building models')
cnn.loadLayers()
train_f = cnn.buildTrainingFunction()
validation_f = cnn.buildValidationFunction()
# Train
prev_errors_train = None
for epoch in range(train_data['last_train_epoch']+1, TRAINING_EPOCHS+1):
  errors = []
  print('* Epoch:', epoch)
  for index in range(max_batch_index):
    b_start = index * MINI_BATCH_SIZE
    b_end = (index+1) * MINI_BATCH_SIZE
    batch_error = train_f(training_in[b_start : b_end], training_out[b_start : b_end])
    errors.append(batch_error)
    if index % error_print_step == 0:
      print('  -', batch_error)
  # Show difference
  errors_train = numpy.asarray(errors)
  if prev_errors_train is not None:
    diff = prev_errors_train - errors_train
    diff_avg = numpy.mean(diff)
    print('=== Average difference:', diff_avg, '===')
    diff_max = numpy.max(diff)
    diff_min = numpy.min(diff)
    print('=== Min/Max difference:', diff_min, '/', diff_max, '===')
  prev_errors_train = errors_train
  # Backup
  if epoch % VALIDATION_STEP == 0 or epoch % BACKUP_STEP == 0:
    print('--- Backup model ---')
    # backup the model
    cnn.saveModel(model_backup_savepath)
    # save training data
    train_data['last_train_epoch'] = epoch
    saveTrainData(train_data, train_status_savepath)
  # Validate
  if epoch % VALIDATION_STEP == 0:
    print('~~~ Validation ~~~')
    valid_error = getValidationError(validation_f, validation_in, validation_out, MINI_BATCH_SIZE)
    print('### Error in validation is', valid_error, '###')
    if train_data['best_validation'] is None or valid_error < train_data['best_validation']:
      print('@@@ Save model @@@')
      train_data['best_validation'] = valid_error
      # save the CNN model
      cnn.saveCNNModel(model_savepath)
      # save train data
      saveTrainData(train_data, train_status_savepath)
print('Fine-tuning complete!')

print('Training complete!')
