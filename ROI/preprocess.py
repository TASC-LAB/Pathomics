##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import csv
import math
import shutil
import datetime

# Third party modules.
import numpy
import imageio
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Local modules.
import inputs

#############################
# Internal Functions.       #
#############################
def _create_dirtree(window, config):
    basepath = r"" + os.path.join(config.get("DEFAULT", "basepath"), "ROI")

    # Create preprocess base dirs.
    config.set("ROI", "processed_path", os.path.join(basepath, "Processed"))
    config.set("ROI", "results_path",   os.path.join(basepath, "Results"))
    os.makedirs(config.get("ROI", "processed_path"), exist_ok=True)
    os.makedirs(config.get("ROI", "results_path"),   exist_ok=True)

    config.set("ROI", "patches_path",  os.path.join(config.get("ROI", "processed_path"), "patches"))
    config.set("ROI", "filtered_path", os.path.join(config.get("ROI", "processed_path"), "filtered"))
    os.makedirs(config.get("ROI", "patches_path"),  exist_ok=True)
    os.makedirs(config.get("ROI", "filtered_path"), exist_ok=True)

    # Create training, test and validation dirs.
    config.set("ROI", "train_path",      os.path.join(config.get("ROI", "processed_path"), "train"))
    config.set("ROI", "test_path",       os.path.join(config.get("ROI", "processed_path"), "test"))
    config.set("ROI", "validation_path", os.path.join(config.get("ROI", "processed_path"), "validation"))
    os.makedirs(config.get("ROI", "train_path"),      exist_ok=True)
    os.makedirs(config.get("ROI", "test_path"),       exist_ok=True)
    os.makedirs(config.get("ROI", "validation_path"), exist_ok=True)

    # Create results dirs.
    config.set("ROI", "models_path", os.path.join(config.get("ROI", "results_path"), "models"))
    os.makedirs(config.get("ROI", "models_path"), exist_ok=True)

    window.processed_path.set(config.get("ROI", "processed_path"))
    window.results_path  .set(config.get("ROI", "results_path"))

def _count_patches(window, config, dataset):
    patches       = 0
    patch_height  = config.getint("ROI", "height")
    patch_width   = config.getint("ROI", "width")
    num_of_images = config.getint("ROI", "num_of_images")
    images_file   = os.path.join(config.get("ROI", "results_path"), "images.csv")
    config.set("ROI", "images_file", images_file)

    window.roi_status.set(f"Counting {num_of_images} images.")
    window.roi_progress.set(0)
    with open(images_file, "w+", newline='') as filehandle:
        image_csv = csv.writer(filehandle, delimiter=',')
        image_csv.writerow(["path", "image number", "height", "width", "patches"])
        for image_index, (path, image) in enumerate(dataset):
            height, width, _ = image.shape
            patches += (height // patch_height) * (width // patch_width)
            window.roi_progress.set(100 * (image_index + 1) / num_of_images)
            image_csv.writerow([path, image_index + 1, height, width, (height // patch_height) * (width // patch_width)])
    config.set("ROI", "patches", str(patches))

def _patch_image(config, image):
    img_height, img_width, _ = image.shape
    height = config.getint("ROI", "height")
    width  = config.getint("ROI", "width")

    for row in range(0, img_height, height):
        if row + height > img_height:
            continue
        for col in range(0, img_width, width):
            if col + width > img_width:
                continue
            # TODO: Move to PIL instead of imageio and use crop (more readable).
            # yield image.crop((col, row, col + width, row + height))
            yield image[row : row + height, col : col + width, :]

def _save_image(path, name, image):
    plt.imsave(os.path.join(path, name), numpy.array(image) / 255)

def _save_patches(window, config, dataset, output_type):
    patches_path  = config.get("ROI",    "patches_path")
    patches       = config.getint("ROI", "patches")
    num_of_images = config.getint("ROI", "num_of_images")
    if num_of_images == patches:
        return None

    patches_file  = os.path.join(config.get("ROI", "results_path"), "patches.csv")
    config.set("ROI", "patches_file", patches_file)

    window.roi_status.set(f"Creating {patches} patches.")
    window.roi_progress.set(0)
    with open(patches_file, "w+", newline='') as filehandle:
        patches_csv = csv.writer(filehandle, delimiter=',')
        patches_csv.writerow(["patch", "original image"])
        patches_count = 0
        for path, image in dataset:
            for patch in _patch_image(config, image):
                patches_count += 1
                name = f"{patches_count}.{output_type}"
                _save_image(patches_path, name, patch)
                patches_csv.writerow([name, path])
                window.roi_progress.set(100 * (patches_count) / patches)


#############################
# API Functions.            #
#############################
def pretrain(window, config, train_percent, test_percent, validation_percent):
    window.roi_status.set("Creating directories.")
    _create_dirtree(window, config)

    path          = config.get("ROI", "dataset_path")
    input_type    = config.get("ROI", "input_type")
    output_type   = config.get("ROI", "output_type")
    num_of_images = inputs.count_dataset(path, input_type)
    config.set("ROI", "num_of_images", str(num_of_images))

    dataset = inputs.gen_dataset(path, input_type)
    _count_patches(window, config, dataset)

    dataset = inputs.gen_dataset(path, input_type)
    _save_patches(window, config, dataset, output_type)
    config.set("ROI", "input_type", output_type)

    # Split dataset into sub-sets.
    total           = config.getint("ROI", "patches")
    train_size      = math.floor(train_percent      * total)
    test_size       = math.floor(test_percent       * total)
    validation_size = math.floor(validation_percent * total)

    # Add remainder.
    train_size += (total - train_size - test_size - validation_size)
    config.set("ROI", "train_size",      str(train_size))
    config.set("ROI", "test_size",       str(test_size))
    config.set("ROI", "validation_size", str(validation_size))