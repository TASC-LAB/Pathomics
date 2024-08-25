##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import shutil
import threading

# Third party modules.
import keras
import imageio
import skimage
import matplotlib.pyplot as plt

# Local modules.
import inputs
from .train      import train
from .preprocess import pretrain, prenormalize
from .results    import generate_results, normalize

#############################
# Internal Functions.       #
#############################
def _get_config(window):
    config = window.config
    if not config.has_section("NORMALIZATION"):
        config.add_section("NORMALIZATION")
    config.set("NORMALIZATION", "dataset_path",       config.get("DEFAULT", "dataset_path"))
    config.set("NORMALIZATION", "input_type",         config.get("DEFAULT", "input_file_type"))
    config.set("NORMALIZATION", "output_type",        config.get("DEFAULT", "output_file_type"))
    config.set("NORMALIZATION", "channels",           config.get("DEFAULT", "num_of_channels"))
    config.set("NORMALIZATION", "width",              config.get("DEFAULT", "width"))
    config.set("NORMALIZATION", "height",             config.get("DEFAULT", "height"))
    config.set("NORMALIZATION", "epochs",             "15")
    config.set("NORMALIZATION", "batch_size",         "3000")
    config.set("NORMALIZATION", "train_percent",      "0.3")
    config.set("NORMALIZATION", "test_percent",       "0.7")
    config.set("NORMALIZATION", "validation_percent", "0.0")

    return config

def _norm_train(window):
    config      = _get_config(window)
    norm_path   = config.get("NORMALIZATION", "test_grayscale_path")
    output_type = config.get("NORMALIZATION", "output_type")
    dataset     = inputs.gen_dataset(norm_path, output_type)

    train_percent      = config.getfloat("NORMALIZATION", "train_percent")
    test_percent       = config.getfloat("NORMALIZATION", "test_percent")
    validation_percent = config.getfloat("NORMALIZATION", "validation_percent")

    pretrain(window, config, train_percent, test_percent, validation_percent)
    train(window, config)
    generate_results(window, config)

    model = keras.models.load_model(config.get("NORMALIZATION", "model_path"))
    config.set("NORMALIZATION", "normalized_path", config.get("NORMALIZATION", "generated_path"))
    normalize(window, config, model, dataset)

    window.norm_status.set(f"Normalization algo saved.")
    config.set("GUI", "btnIntensity_state", "normal")
    window.refresh_buttons()

    # Save configuration under dataset and delete thread.
    with window.mutex, open(config.get("DEFAULT", "configfile"), "w+") as filehandle:
        config.write(filehandle)
    del window.threads["norm_train"]

def _norm_load(window):
    config = _get_config(window)
    model_path = inputs.select_file(msg="Please select the model to load.", init_dir=window.assets, types=(("Keras model", "*.keras"), ("DEPRACATED", "*.h5"), ("All files.", "*.*")))
    config.set("NORMALIZATION", "model_path", model_path)
    config.set("NORMALIZATION", "algo_name",  os.path.splitext(os.path.basename(model_path))[0])

    window.norm_algo_name.set(config.get("NORMALIZATION", "algo_name"))
    window.norm_status.set(f"Ready for Normalization.")
    config.set("GUI", "btnNormalize_state", "normal")
    window.refresh_buttons()

    # Save configuration under dataset and delete thread.
    with window.mutex, open(config.get("DEFAULT", "configfile"), "w+") as filehandle:
        config.write(filehandle)
    del window.threads["norm_load"]

def _norm(window):
    config = _get_config(window)

    norm_path  = config.get("NORMALIZATION", "dataset_path")
    input_type = config.get("NORMALIZATION", "input_type")
    dataset    = inputs.gen_dataset(norm_path, input_type)
    model      = keras.models.load_model(config.get("NORMALIZATION", "model_path"))

    pretrain(window, config, 0, 1, 0)
    normalize(window, config, model, dataset)
    config.set("DEFAULT", "dataset_path",     config.get("NORMALIZATION", "normalized_path"))
    config.set("DEFAULT", "input_file_type",  config.get("NORMALIZATION", "input_type"))
    config.set("DEFAULT", "output_file_type", config.get("NORMALIZATION", "output_type"))

    config.set("GUI", "btnIntensity_state", "normal")
    window.pathomics_status.set("Ready for QuPath.")
    window.norm_status.set("Done Normalizing.")
    window.refresh_buttons()

    # Save configuration under dataset and delete thread.
    with window.mutex, open(config.get("DEFAULT", "configfile"), "w+") as filehandle:
        config.write(filehandle)
    del window.threads["norm"]

def _intensity_dist(window):
    config = _get_config(window)
    output_type     = config.get("NORMALIZATION", "output_type")
    normalized_path = config.get("NORMALIZATION", "normalized_path")
    results_path    = config.get("NORMALIZATION", "results_path")
    channels        = config.getint("NORMALIZATION", "channels")

    path  = inputs.select_file(msg="Please select image for Intensity Distribution analysis.", init_dir=normalized_path, types=((output_type.title(), '*.'+output_type), ('All files', '*.*')))
    image = imageio.imread(path)

    plt.switch_backend('Agg')
    fig, axes = plt.subplots(nrows=channels, ncols=1, figsize=(10, 10))
    for index, color in enumerate(['red', 'green', 'blue']):
        img_hist, bins = skimage.exposure.histogram(image[..., index], source_range='dtype')
        axes[index].plot(bins, img_hist / img_hist.max(), label='Intensity Histogram')
        img_cdf, bins = skimage.exposure.cumulative_distribution(image[..., index])
        axes[index].plot(bins, img_cdf, label='Cumulative Distribution Function')
        axes[index].set_ylabel(color)
        axes[index].legend(loc='upper left')
        axes[index].set_title(os.path.basename(path))
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "IntensityDistribution.jpg"))
    plt.switch_backend('TkAgg')

    # TODO: Add intensity distribution logic.
    window.refresh_buttons()

    # Save configuration under dataset and delete thread.
    with window.mutex, open(config.get("DEFAULT", "configfile"), "w+") as filehandle:
        config.write(filehandle)
    del window.threads["intensity_dist"]

#############################
# API Functions.            #
#############################
def norm_train(window):
    if "norm_train" not in window.threads:
        window.threads["norm_train"] = threading.Thread(target=_norm_train, args=(window,)).start()

def norm_load(window):
    if "norm_load" not in window.threads:
        window.threads["norm_load"] = threading.Thread(target=_norm_load, args=(window,)).start()

def norm(window):
    if "norm" not in window.threads:
        window.threads["norm"] = threading.Thread(target=_norm, args=(window,)).start()

def intensity_dist(window):
    if "intensity_dist" not in window.threads:
        window.threads["intensity_dist"] = threading.Thread(target=_intensity_dist, args=(window,)).start()