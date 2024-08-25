##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import csv

# Third party modules.
import numpy
import imageio

# Local modules.
import inputs

#############################
# Internal Functions.       #
#############################
def _hardcode_filter(config, image):
    white_threshold = config.getint("ROI", "white_threshold")
    black_threshold = config.getint("ROI", "black_threshold")
    res = numpy.mean(image)
    return res < white_threshold and res > black_threshold

#############################
# API Functions.            #
#############################
def ROIfilter(window, config):
    if config.has_option("ROI", "filtered"):
        return None
    config.set("ROI", "filtered", "True")

    path           = config.get("ROI", "patches_path")
    output_type    = config.get("ROI", "output_type")
    filtered_path  = config.get("ROI", "filtered_path")
    num_of_patches = config.getint("ROI", "patches")
    dataset        = inputs.gen_dataset(path, output_type)
    filtered_file  = os.path.join(config.get("ROI", "results_path"), "filtered.csv")

    window.roi_status.set(f"Filtering {num_of_patches} images.")
    window.roi_progress.set(0)
    with open(filtered_file, "w+", newline='') as filehandle:
        filtered_csv = csv.writer(filehandle, delimiter=',')
        filtered_csv.writerow(["patch", "patch number"])
        for patch_index, (path, image) in enumerate(dataset):
            if _hardcode_filter(config, image):
                imageio.imwrite(os.path.join(filtered_path, f"{patch_index + 1}.{output_type}"), image)
                filtered_csv.writerow([path, patch_index + 1])
            window.roi_progress.set(100 * (patch_index + 1) / num_of_patches)