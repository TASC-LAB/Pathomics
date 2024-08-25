#!/bin/python3
import os
import shutil
import argparse

import logic
import Models
import inputs
import outputs
import preprocess

##############################
# ARGUMENTS & PARAMETERS     #
##############################
parser = argparse.ArgumentParser(description="Normalize a dataset of images.")

# Preprocess arguments.
parser.add_argument("--dataset",       action="store",      required=True,  type=str,                       help="Path to the original dataset.")
parser.add_argument("--input_format",  action="store",      default="tiff", choices=["scn", "png", "tiff"], help="Input images format.")
parser.add_argument("--output_format", action="store",      default="tiff", choices=["scn", "png", "tiff"], help="Output format to save.")
parser.add_argument("--clear",         action="store_true", default=False,  dest="clear",                   help="Clear all preprocessed files.")
parser.add_argument("--use_filter",    action="store_true", default=False,                                  help="Preprocess all images without the filter function.")
parser.add_argument("--preprocessed",  action="store_true", default=False,                                  help="Preprocess the dataset into color, grayscale & combined and then split into train, test and validation.")

# Image arguments.
parser.add_argument("--image_width",  action="store", default=256, type=int, dest="width",  help="Width of the final images.")
parser.add_argument("--image_height", action="store", default=256, type=int, dest="height", help="Height of the final images.")
parser.add_argument("--channels",     action="store", default=3,   type=int,                help="Number of channels in the images.")

# Training arguments.
parser.add_argument("--epochs",     action="store", default=15,   type=int,                      help="Number of epochs, in each train the model with a random batch of images.")
parser.add_argument("--train",      action="store", default=0.3,  type=float,                    help="Percentage of the images that go into training.")
parser.add_argument("--test",       action="store", default=0.7,  type=float,                    help="Percentage of the images that go into testing.")
parser.add_argument("--val",        action="store", default=0,    type=float, dest="validation", help="Percentage of the images that go into validation.")
parser.add_argument("--batch_size", action="store", default=3000, type=int,   dest="batch_size", help="Minimum batch size for an epoch.")

args = parser.parse_args()

# Create directories.
processed_path = os.path.join(args.dataset, "Processed")
results_path   = os.path.join(args.dataset, "Results")

# Compile the parameters into compact variables.
shape = (args.height, args.width, args.channels)
split = (args.train,  args.test,  args.validation)

##############################
# INPUT VALIDATION           #
##############################
if not os.path.exists(args.dataset):
    print(f"The dataset path ({args.dataset}) does not exists.")
    exit()

# Normalize the percentages to 100%
total_weight = args.train + args.test + args.validation
if total_weight != 1:
    args.train      /= total_weight
    args.test       /= total_weight
    args.validation /= total_weight

if args.width <= 0 or args.height <= 0:
    print(f"Image dimensions are non-positive: {args.width}x{args.height}")
    exit()

if args.channels <= 0:
    print(f"Invalid number of channels: {args.channels}")
    exit()

if args.epochs <= 0:
    print(f"Invalid number of epochs: {args.epochs}")
    exit()

if args.batch_size <= 0:
    print(f"Invalid batch size: {args.batch_size}")
    exit()

if args.clear:
    shutil.rmtree(processed_path, ignore_errors=True)
    shutil.rmtree(results_path,   ignore_errors=True)

##############################
# CREATE DIRECTORIES         #
##############################
os.makedirs(processed_path,                            exist_ok=True)
os.makedirs(os.path.join(processed_path, "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "combined"),  exist_ok=True)

os.makedirs(os.path.join(processed_path, "train", "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "train", "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "train", "combined"),  exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",  "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",  "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",  "combined"),  exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",   "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",   "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",   "combined"),  exist_ok=True)

os.makedirs(results_path,                         exist_ok=True)
os.makedirs(os.path.join(results_path, "models"), exist_ok=True)

##############################
# PREPROCESS DATASETS        #
##############################
if not args.preprocessed:
    # Generate basic processed dataset.
    print(f"Splitting original images into {args.width}x{args.height} (color, grayscale & combined).")
    size = 0
    for i, image in enumerate(inputs.load_dataset(args.dataset, suffix=args.input_format)):
        print(f"Image #{i}: {image.shape} | type: {image.dtype}")
        color, grayscale, combined = preprocess.process_image(image, shape, args.use_filter)
        outputs.save_dataset(os.path.join(processed_path,         "color"),     color,     offset=size, suffix=args.output_format)
        outputs.save_dataset(os.path.join(processed_path,         "grayscale"), grayscale, offset=size, suffix=args.output_format)
        size += outputs.save_dataset(os.path.join(processed_path, "combined"),  combined,  offset=size, suffix=args.output_format)

    # Load dataset generators.
    print("Loading color & grayscale datasets.")
    color     = inputs.load_dataset(os.path.join(processed_path, "color"),     suffix=args.output_format)
    grayscale = inputs.load_dataset(os.path.join(processed_path, "grayscale"), suffix=args.output_format)
    combined  = inputs.load_dataset(os.path.join(processed_path, "combined"),  suffix=args.output_format)

    # Create sup-datasets.
    print(f"Splitting datasets into train ({args.train*100}%), test ({args.test*100}%) and validation ({args.validation*100}%).")
    train, test, val = preprocess.split_dataset((color, grayscale, combined), size, split)

    # Save processed datasets.
    color, grayscale, combined = train
    outputs.save_dataset(os.path.join(processed_path, "train", "color"),     color,     suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "train", "grayscale"), grayscale, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "train", "combined"),  combined,  suffix=args.output_format)
    color, grayscale, combined = test
    outputs.save_dataset(os.path.join(processed_path, "test", "color"),     color,     suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "test", "grayscale"), grayscale, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "test", "combined"),  combined,  suffix=args.output_format)
    color, grayscale, combined = val
    outputs.save_dataset(os.path.join(processed_path, "val", "color"),     color,     suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "val", "grayscale"), grayscale, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "val", "combined"),  combined,  suffix=args.output_format)

##############################
# MAIN LOGIC                 #
##############################
training_path = os.path.join(processed_path, "train", "combined")
batch_size    = min(args.batch_size, inputs.get_size(training_path, suffix=args.output_format))
models        = Models.define_models(shape)

# Optimization - not enough images for multiple epochs.
if batch_size < args.batch_size:
    args.epochs = 1

# Train & test the GAN model.
print("Starting to train models.")
for epoch in range(1, args.epochs + 1):
    # Split the dataset into equal random batches.
    # Convert the image into numpy arrays.
    batch = inputs.load_batch(training_path, batch_size, suffix=args.output_format)

    # Train the model and calculate losses.
    print(f"Epoch #{epoch} | Batch size: {batch_size}")
    models, losses = logic.train(results_path, models, batch, epoch)

    # Save the losses in the results directory.
    outputs.save_losses(results_path, losses, epoch)

print("Starting best model test.")
model, epoch = inputs.get_best_model(results_path)
dataset      = inputs.load_dataset(os.path.join(processed_path, "test", "grayscale"), suffix=args.output_format)
for index, grayscale in enumerate(dataset):
    grayscale = inputs.convert(grayscale)
    generated = logic.normalize(model, grayscale)
    generated = outputs.convert(generated)
    outputs.save_image(os.path.join(results_path, "generated"), generated, offset=index + 1, suffix=args.output_format)

# Outputs of the models.
outputs.plot_outputs(results_path, epoch, batch_size)
print("Done.")