##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import csv
import math
import shutil
import random
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
    basepath = r"" + os.path.join(config.get("DEFAULT", "basepath"), "Normalization")

    # Create preprocess base dirs: color, grayscale, combined.
    config.set("NORMALIZATION", "processed_path", os.path.join(basepath, "Processed"))
    config.set("NORMALIZATION", "results_path",   os.path.join(basepath, "Results"))
    os.makedirs(config.get("NORMALIZATION", "processed_path"), exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "results_path"),   exist_ok=True)

    config.set("NORMALIZATION", "color_path",     os.path.join(config.get("NORMALIZATION", "processed_path"), "color"))
    config.set("NORMALIZATION", "grayscale_path", os.path.join(config.get("NORMALIZATION", "processed_path"), "grayscale"))
    config.set("NORMALIZATION", "combined_path",  os.path.join(config.get("NORMALIZATION", "processed_path"), "combined"))
    os.makedirs(config.get("NORMALIZATION", "color_path"),     exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "grayscale_path"), exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "combined_path"),  exist_ok=True)

    # Create training, test and validation dirs.
    config.set("NORMALIZATION", "train_path",      os.path.join(config.get("NORMALIZATION", "processed_path"), "train"))
    config.set("NORMALIZATION", "test_path",       os.path.join(config.get("NORMALIZATION", "processed_path"), "test"))
    config.set("NORMALIZATION", "validation_path", os.path.join(config.get("NORMALIZATION", "processed_path"), "validation"))
    os.makedirs(config.get("NORMALIZATION", "train_path"),      exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "test_path"),       exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "validation_path"), exist_ok=True)

    config.set("NORMALIZATION", "train_color_path",     os.path.join(config.get("NORMALIZATION", "train_path"), "color"))
    config.set("NORMALIZATION", "train_grayscale_path", os.path.join(config.get("NORMALIZATION", "train_path"), "grayscale"))
    config.set("NORMALIZATION", "train_combined_path",  os.path.join(config.get("NORMALIZATION", "train_path"), "combined"))
    os.makedirs(config.get("NORMALIZATION", "train_color_path"),     exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "train_grayscale_path"), exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "train_combined_path"),  exist_ok=True)

    config.set("NORMALIZATION", "test_color_path",     os.path.join(config.get("NORMALIZATION", "test_path"), "color"))
    config.set("NORMALIZATION", "test_grayscale_path", os.path.join(config.get("NORMALIZATION", "test_path"), "grayscale"))
    config.set("NORMALIZATION", "test_combined_path",  os.path.join(config.get("NORMALIZATION", "test_path"), "combined"))
    os.makedirs(config.get("NORMALIZATION", "test_color_path"),     exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "test_grayscale_path"), exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "test_combined_path"),  exist_ok=True)

    config.set("NORMALIZATION", "validation_color_path",     os.path.join(config.get("NORMALIZATION", "validation_path"), "color"))
    config.set("NORMALIZATION", "validation_grayscale_path", os.path.join(config.get("NORMALIZATION", "validation_path"), "grayscale"))
    config.set("NORMALIZATION", "validation_combined_path",  os.path.join(config.get("NORMALIZATION", "validation_path"), "combined"))
    os.makedirs(config.get("NORMALIZATION", "validation_color_path"),     exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "validation_grayscale_path"), exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "validation_combined_path"),  exist_ok=True)

    # Create results dirs.
    config.set("NORMALIZATION", "models_path",     os.path.join(config.get("NORMALIZATION", "results_path"), "models"))
    config.set("NORMALIZATION", "generated_path",  os.path.join(config.get("NORMALIZATION", "results_path"), "generated"))
    config.set("NORMALIZATION", "normalized_path", os.path.join(config.get("NORMALIZATION", "results_path"), "normalized"))
    os.makedirs(config.get("NORMALIZATION", "models_path"),     exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "generated_path"),  exist_ok=True)
    os.makedirs(config.get("NORMALIZATION", "normalized_path"), exist_ok=True)

    window.processed_path.set(config.get("NORMALIZATION", "processed_path"))
    window.results_path  .set(config.get("NORMALIZATION", "results_path"))

def _count_patches(window, config, dataset):
    patches       = 0
    patch_height  = config.getint("NORMALIZATION", "height")
    patch_width   = config.getint("NORMALIZATION", "width")
    num_of_images = config.getint("NORMALIZATION", "num_of_images")
    images_file   = os.path.join(config.get("NORMALIZATION", "results_path"), "images.csv")
    config.set("NORMALIZATION", "images_file", images_file)

    window.norm_status.set(f"Counting {num_of_images} images.")
    window.norm_progress.set(0)
    with open(images_file, "w+", newline='') as filehandle:
        image_csv = csv.writer(filehandle, delimiter=',')
        image_csv.writerow(["path", "image number", "height", "width", "patches"])
        for image_index, (path, image) in enumerate(dataset):
            height, width, _ = image.shape
            patches += (height // patch_height) * (width // patch_width)
            window.norm_progress.set(100 * (image_index + 1) / num_of_images)
            image_csv.writerow([path, image_index + 1, height, width, (height // patch_height) * (width // patch_width)])
    config.set("NORMALIZATION", "patches", str(patches))

def _patch_image(config, image):
    img_height, img_width, _ = image.shape
    height = config.getint("NORMALIZATION", "height")
    width  = config.getint("NORMALIZATION", "width")

    for row in range(0, img_height, height):
        if row + height > img_height:
            continue
        for col in range(0, img_width, width):
            if col + width > img_width:
                continue
            # TODO: Move to PIL instead of imageio and use crop (more readable).
            # yield image.crop((col, row, col + width, row + height))
            yield image[row : row + height, col : col + width, :]

def _convert_grayscale(color):
    # TODO: PIL conversion might be the best option.
    # Image.open(<path>).convert('LA')
    grayscale = numpy.dot(color[...,:3], [0.299, 0.587, 0.144])
    grayscale = numpy.round(grayscale).astype(numpy.uint8)
    grayscale = numpy.stack([grayscale] * 3, axis=-1)
    return grayscale

def _combine(color, grayscale):
    combined = numpy.hstack((color, grayscale))
    return combined

def _save_image(path, name, image):
    plt.imsave(os.path.join(path, name), numpy.array(image) / 255)

def _save_patches(window, config, dataset, output_type):
    color_path     = config.get("NORMALIZATION",    "color_path")
    grayscale_path = config.get("NORMALIZATION",    "grayscale_path")
    combined_path  = config.get("NORMALIZATION",    "combined_path")
    patches        = config.getint("NORMALIZATION", "patches")
    num_of_images  = config.getint("NORMALIZATION", "num_of_images")
    if num_of_images == patches:
        dataset_path = config.get("NORMALIZATION", "dataset_path")
        config.set("NORMALIZATION", "color_path",     dataset_path)
        config.set("NORMALIZATION", "grayscale_path", dataset_path)
        config.set("NORMALIZATION", "combined_path",  dataset_path)
        return None

    patches_file   = os.path.join(config.get("NORMALIZATION", "results_path"), "patches.csv")
    config.set("NORMALIZATION", "patches_file", patches_file)

    window.norm_status.set(f"Creating {patches} patches.")
    window.norm_progress.set(0)
    with open(patches_file, "w+", newline='') as filehandle:
        patches_csv = csv.writer(filehandle, delimiter=',')
        patches_csv.writerow(["patch", "original image", "color", "grayscale", "combined"])
        patches_count = 0
        for path, image in dataset:
            for patch in _patch_image(config, image):
                patches_count += 1

                color     = patch
                grayscale = _convert_grayscale(color)
                combined  = _combine(color, grayscale)

                name = f"{patches_count}.{output_type}"
                _save_image(color_path,     name, color)
                _save_image(grayscale_path, name, grayscale)
                _save_image(combined_path,  name, combined)

                patches_csv.writerow([name, path, "True", "True", "True"])
                window.norm_progress.set(100 * (patches_count) / patches)

def _create_subsets(window, config):
    if config.has_option("NORMALIZATION", "subsets"):
        return None
    config.set("NORMALIZATION", "subsets", "True")

    window.norm_status.set(f"Creating subsets.")
    output_type     = config.get("NORMALIZATION",    "output_type")
    color_path      = config.get("NORMALIZATION",    "color_path")
    grayscale_path  = config.get("NORMALIZATION",    "grayscale_path")
    combined_path   = config.get("NORMALIZATION",    "combined_path")
    train_size      = config.getint("NORMALIZATION", "train_size")
    test_size       = config.getint("NORMALIZATION", "test_size")
    validation_size = config.getint("NORMALIZATION", "validation_size")
    total           = config.getint("NORMALIZATION", "patches")

    # List of all the file names.
    patches_list = list(range(1, total + 1))
    patches_list = list(map(str, patches_list))
    random.shuffle(patches_list)

    # Generate 'train' data subset.
    train_color_path     = config.get("NORMALIZATION", "train_color_path")
    train_grayscale_path = config.get("NORMALIZATION", "train_grayscale_path")
    train_combined_path  = config.get("NORMALIZATION", "train_combined_path")
    for index in range(train_size):
        src = f"{patches_list[0]}.{output_type}"
        dst = f"{str(index + 1)}.{output_type}"
        shutil.copyfile(os.path.join(color_path,     src), os.path.join(train_color_path,     dst))
        shutil.copyfile(os.path.join(grayscale_path, src), os.path.join(train_grayscale_path, dst))
        shutil.copyfile(os.path.join(combined_path,  src), os.path.join(train_combined_path,  dst))
        patches_list.pop(0)
        window.norm_progress.set(100 * (index + 1) / total)

    # Generate 'test' data subset.
    test_color_path     = config.get("NORMALIZATION", "test_color_path")
    test_grayscale_path = config.get("NORMALIZATION", "test_grayscale_path")
    test_combined_path  = config.get("NORMALIZATION", "test_combined_path")
    for index in range(test_size):
        src = f"{patches_list[0]}.{output_type}"
        dst = f"{str(index + 1)}.{output_type}"
        shutil.copyfile(os.path.join(color_path,     src), os.path.join(test_color_path,     dst))
        shutil.copyfile(os.path.join(grayscale_path, src), os.path.join(test_grayscale_path, dst))
        shutil.copyfile(os.path.join(combined_path,  src), os.path.join(test_combined_path,  dst))
        patches_list.pop(0)
        window.norm_progress.set(100 * (train_size + index + 1) / total)

    # Generate 'validation' data subset.
    validation_color_path     = config.get("NORMALIZATION", "validation_color_path")
    validation_grayscale_path = config.get("NORMALIZATION", "validation_grayscale_path")
    validation_combined_path  = config.get("NORMALIZATION", "validation_combined_path")
    for index in range(validation_size):
        src = f"{patches_list[0]}.{output_type}"
        dst = f"{str(index + 1)}.{output_type}"
        shutil.copyfile(os.path.join(color_path,     src), os.path.join(validation_color_path,     dst))
        shutil.copyfile(os.path.join(grayscale_path, src), os.path.join(validation_grayscale_path, dst))
        shutil.copyfile(os.path.join(combined_path,  src), os.path.join(validation_combined_path,  dst))
        patches_list.pop(0)
        window.norm_progress.set(100 * (train_size + test_size + index + 1) / total)

#############################
# API Functions.            #
#############################
# Convert RGB image with unsigned byte values to float values from -1 to 1.
def image2train(image):
    image = numpy.array(image) / 127.5 - 1.
    return numpy.reshape(image, tuple([1] + list(image.shape)))

# Convert float array from -1 to 1 values to an RGB image with unsigned bytes.
def train2image(image):
    image = numpy.reshape(image, tuple(list(image.shape)[1:]))
    return ((image + 1.) * 127.5).astype(numpy.uint8)

def prenormalize(window, config, dataset):
    for image in dataset:
        yield from _convert_grayscale(image)

def pretrain(window, config, train_percent, test_percent, validation_percent):
    window.norm_status.set("Creating directories.")
    _create_dirtree(window, config)

    path          = config.get("NORMALIZATION", "dataset_path")
    input_type    = config.get("NORMALIZATION", "input_type")
    output_type   = config.get("NORMALIZATION", "output_type")
    num_of_images = inputs.count_dataset(path, input_type)
    config.set("NORMALIZATION", "num_of_images", str(num_of_images))

    dataset = inputs.gen_dataset(path, input_type)
    _count_patches(window, config, dataset)

    dataset = inputs.gen_dataset(path, input_type)
    _save_patches(window, config, dataset, output_type)
    config.set("NORMALIZATION", "input_type", output_type)

    # Split dataset into sub-sets.
    total           = config.getint("NORMALIZATION", "patches")
    batch_size      = config.getint("NORMALIZATION", "batch_size")
    train_size      = math.floor(train_percent      * total)
    test_size       = math.floor(test_percent       * total)
    validation_size = math.floor(validation_percent * total)

    # Add remainder.
    train_size += (total - train_size - test_size - validation_size)

    config.set("NORMALIZATION", "train_size",      str(train_size))
    config.set("NORMALIZATION", "test_size",       str(test_size))
    config.set("NORMALIZATION", "validation_size", str(validation_size))
    config.set("NORMALIZATION", "batch_size",      str(min(train_size, batch_size)))
    config.set("NORMALIZATION", "norm_size",       str(test_size))
    _create_subsets(window, config)