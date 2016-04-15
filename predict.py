#!/usr/bin/env python3

import os
import sys

path = os.path.abspath(__file__)
MAIN_FOLDER = os.path.dirname(path)

import argparse
from cnn.GrainRemover import GrainRemover

parser = argparse.ArgumentParser(description="Grain remover")
parser.add_argument('--input', '-i', dest='input', default=None,
                    help='Input image')
parser.add_argument('--output', '-o', dest='output', default=None,
                    help='Output image')
parser.add_argument('--model', '-m', dest='model', default=None,
                    help='Model file')
args = parser.parse_args()

input_path = args.input
output_path = args.output
model_path = args.model

# Check inputs - required
if input_path is None or not os.path.exists(input_path):
  print('Input image not found.')
  sys.exit(1)
elif output_path is None:
  print('Output savepath missing.')
  sys.exit(1)

# Check inputs - optional
if model_path is None:
  model_path = os.path.join(MAIN_FOLDER, 'models/model_b.json')

if not os.path.exists(model_path):
  print('Warning! Model not found.')
  sys.exit(1)

# Load network
cnn = GrainRemover()

cnn.loadCNNModelFromFile(model_path)

# Convert image to Data
img = cnn.convertInputImageToData(input_path)

# Predict
predict = cnn.buildPredictFunction()
rmod = predict(img)

# Save result
rimg = cnn.convertDataToImage(rmod)

rimg.save(output_path)

