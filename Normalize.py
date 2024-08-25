#!/bin/python3
import os
import argparse

import numpy
from keras.models import load_model
from PIL import Image

import logic
import inputs
import outputs

##############################
# ARGUMENTS & PARAMETERS     #
##############################
parser = argparse.ArgumentParser(description="Normalize a dataset of images.")
parser.add_argument("--model",         action="store", dest="model",    type=str, required=True,  help="Path to the model for normalization.")
parser.add_argument("--dataset",       action="store", dest="dataset",  type=str, required=True,  help="Path to the dataset to normalize.")
parser.add_argument("--results",       action="store", dest="results",  type=str, required=False, help="Path to store the normalized dataset.")
parser.add_argument("--channels",      action="store", dest="channels", type=int, default=3,      help="Number of channels in the images.")
parser.add_argument("--image_width",   action="store", dest="width",    type=int, default=256,    help="Width of the images in the dataset.")
parser.add_argument("--image_height",  action="store", dest="height",   type=int, default=256,    help="Height of the images in the dataset.")
parser.add_argument("--input_format",  action="store", default="scn", choices=["scn", "png", "tiff"], help="Input images format.")
parser.add_argument("--output_format", action="store", default="scn", choices=["scn", "png", "tiff"], help="Input images format.")
args = parser.parse_args()

##############################
# INPUT VALIDATION           #
##############################
if not os.path.exists(args.model):
    print(f"The model path ({args.model}) does not exists.")
    exit()

if not os.path.exists(args.dataset):
    print(f"The dataset path ({args.dataset}) does not exists.")
    exit()

if "results" not in args:
    args.results = os.path.join(args.dataset, "Results", "normalized")

##############################
# MAIN LOGIC                 #
##############################
model   = load_model(args.model)
dataset = inputs.load_dataset(args.dataset, args.input_format)
print("Starting")
for index, grayscale in enumerate(dataset):
    grayscale = inputs.convert(grayscale)
    generated = logic.normalize(model, grayscale)
    generated = outputs.convert(generated)
    outputs.save_image(args.results, generated, offset=index + 1, suffix=args.output_format)
print("Done.")