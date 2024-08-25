##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import threading

# Local modules.
import inputs
from .preprocess import pretrain
from .results    import ROIfilter

#############################
# Internal Functions.       #
#############################
def _get_config(window):
    config = window.config
    if not config.has_section("ROI"):
        config.add_section("ROI")
    config.set("ROI", "dataset_path",       config.get("DEFAULT", "dataset_path"))
    config.set("ROI", "input_type",         config.get("DEFAULT", "input_file_type"))
    config.set("ROI", "output_type",        config.get("DEFAULT", "output_file_type"))
    config.set("ROI", "channels",           config.get("DEFAULT", "num_of_channels"))
    config.set("ROI", "width",              config.get("DEFAULT", "width"))
    config.set("ROI", "height",             config.get("DEFAULT", "height"))
    config.set("ROI", "train_percent",      "0.0")
    config.set("ROI", "test_percent",       "1.0")
    config.set("ROI", "validation_percent", "0.0")
    config.set("ROI", "white_threshold",    "192")
    config.set("ROI", "black_threshold",    "64")

    return config

def _roi_train(window, params, dataset):
    pass

def _roi_load(window, params):
    pass

def _roi(window):
    config = _get_config(window)

    pretrain(window, config, 0, 1, 0)
    ROIfilter(window, config)
    config.set("DEFAULT", "dataset_path",     config.get("ROI", "filtered_path"))
    config.set("DEFAULT", "input_file_type",  config.get("ROI", "input_type"))
    config.set("DEFAULT", "output_file_type", config.get("ROI", "output_type"))

    window.pathomics_status.set("Ready for ROI.")
    window.roi_status.set("Done Filtering.")
    window.refresh_buttons()

    # Save configuration under dataset and delete thread.
    with window.mutex, open(config.get("DEFAULT", "configfile"), "w+") as filehandle:
        config.write(filehandle)
    del window.threads["roi"]

#############################
# API Functions.            #
#############################
def roi_train(window):
    pass
    # if "roi_train" not in window.threads:
    #     window.roi_params = ROIParams(window.state.dataset_path.get(), window.state.input_file_type.get(), window.state.output_file_type.get())
    #     window.threads["roi_train"] = threading.Thread(target=_roi_train, args=(window, window.roi_params, window.state.dataset)).start()

def roi_load(window):
    pass
    # if "roi_load" not in window.threads:
    #     window.roi_params = ROIParams(window.state.dataset_path.get(), window.state.input_file_type.get(), window.state.output_file_type.get())
    #     window.threads["roi_load"] = threading.Thread(target=_roi_load, args=(window, window.roi_params)).start()

def roi(window):
    if "roi" not in window.threads:
        window.threads["roi"] = threading.Thread(target=_roi, args=(window,)).start()