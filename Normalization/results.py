##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import glob

# Third party modules.
import keras
import numpy
import imageio
import matplotlib
import matplotlib.pyplot as plt

# Local modules.
import inputs
from .preprocess import image2train, train2image

#############################
# Class Definitions.        #
#############################
class NormalizationResults():
    def __init__(self, epoch):
        self.epoch = epoch
        self.disc_fake_loss = []
        self.disc_real_loss = []
        self.generator_loss = []
        self.start_time = None
        self.end_time   = None

#############################
# Internal Functions.       #
#############################
def _smooth_curve(points, factor):
    smoothed_points = [points[0]]
    for point in points[1:]:
        smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
    return smoothed_points

def _select_best_model(config):
    results_path = config.get("NORMALIZATION", "results_path")
    models_path  = config.get("NORMALIZATION", "models_path")

    min_epoch = 1
    min_loss  = numpy.load(os.path.join(results_path, "E1_Generator_loss.npy"))

    for array in glob.glob(os.path.join(results_path, "E*_Generator_loss.npy")):
        epoch = int(array.split('E')[1].split('_')[0])
        loss  = numpy.load(array)

        if numpy.min(loss) < numpy.min(min_loss):
            min_epoch = epoch
            min_loss  = loss

    model_path = os.path.join(models_path, f"g_model_{min_epoch}.keras")
    config.set("NORMALIZATION", "model_path", model_path)

    result = NormalizationResults(min_epoch)
    result.disc_fake_loss = numpy.load(os.path.join(results_path, f"E{min_epoch}_Discriminator_fake_loss.npy"))
    result.disc_real_loss = numpy.load(os.path.join(results_path, f"E{min_epoch}_Discriminator_real_loss.npy"))
    result.generator_loss = numpy.load(os.path.join(results_path, f"E{min_epoch}_Generator_loss.npy"))
    return result

def _save_results(window, config, results):
    results_path = config.get("NORMALIZATION", "results_path")
    config.set("NORMALIZATION", "best_model_epoch", str(results.epoch))

    samples = range(0, len(results.disc_real_loss))
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(samples, _smooth_curve(results.disc_real_loss, factor=0.0), linewidth=0.8, color='#9b0000', marker='.')
    plt.title('Discriminator real loss')
    plt.xlabel('samples')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_path, "Discriminator_real_loss.jpg"))
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(samples, _smooth_curve(results.disc_fake_loss, factor=0.3), linewidth=0.8, color='#1562ff', marker='.')
    plt.title('Discriminator fake loss')
    plt.xlabel('samples')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_path, "Discriminator_fake_loss.jpg"))
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(samples, _smooth_curve(results.generator_loss, factor=0.6), linewidth=0.8, color='#8a3ac6', marker='.')
    plt.title('Generator loss')
    plt.xlabel('samples')
    plt.ylabel('loss')
    plt.savefig(os.path.join(results_path, "Generator_loss.jpg"))
    plt.switch_backend('TkAgg')

#############################
# API Functions.            #
#############################
def normalize(window, config, model, dataset):
    if config.has_option("NORMALIZATION", "normalized"):
        return None
    config.set("NORMALIZATION", "normalized", "True")

    normalized_path = config.get("NORMALIZATION", "normalized_path")
    output_type     = config.get("NORMALIZATION", "output_type")
    norm_size       = config.getint("NORMALIZATION", "norm_size")

    window.norm_status.set(f"Normalizing {norm_size} images.")
    window.norm_progress.set(0)
    for index, (path, image) in enumerate(dataset):
        image = image2train(image)
        image = model.predict(image)
        image = train2image(image)
        imageio.imwrite(os.path.join(normalized_path, f"{index + 1}.{output_type}"), image)
        window.norm_progress.set(100 * (index + 1) / norm_size)

def generate_results(window, config):
    results = _select_best_model(config)
    _save_results(window, config, results)