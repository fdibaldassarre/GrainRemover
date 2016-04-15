#!/usr/bin/env python3

import os
import sys
import numpy
from PIL import Image

path = os.path.abspath(__file__)
MAIN_FOLDER = os.path.dirname(path)

import argparse
from cnn.GrainRemover import GrainRemover

def convertImageToData(imgpath):
  img = Image.open(imgpath)
  img = img.convert('L')
  img = numpy.asarray(img) / 255.
  return img

# Load Extractor
cnn = GrainRemover()

model_savepath = os.path.join(MAIN_FOLDER, 'models/model_train.json')
if not os.path.exists(model_savepath):
  print('Model', model_savepath, 'not found!')
  sys.exit(1)
  #cnn.initializeModel()
else:
  print('Loading model from file')
  cnn.loadModelFromFile(model_savepath)

test_folder = os.path.join(MAIN_FOLDER, 'data/images/test/noise')
test_clean_folder = os.path.join(MAIN_FOLDER, 'data/images/test/clean')

print('Build model')
predict = cnn.buildPredictFunction()

print('Test')
for image_name in os.listdir(test_folder):
  input_path = os.path.join(test_folder, image_name)
  clean_path = os.path.join(test_clean_folder, image_name)
  
  # Convert image to Data
  img = cnn.convertInputImageToData(input_path)

  # Predict
  result = predict(img)
  
  # Confront
  noise = convertImageToData(input_path)
  clean = convertImageToData(clean_path)
  
  diff_base = numpy.sum((noise-clean)**2)
  diff_ml = numpy.sum((result-clean)**2)
  
  accuracy = (diff_base - diff_ml) / diff_base * 100.
  
  print(image_name, accuracy, '%')
  # Save result
  #rimg = cnn.convertDataToImage(rmod)

  #rimg.save(output_path)

