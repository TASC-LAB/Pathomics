##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import json
import threading

# Local modules.
import inputs
from .analyze import label_images

#############################
# Internal Functions.       #
#############################
def _qupath_analyze(window):
    config = window.config
    if not config.has_section("QUPATH"):
        config.add_section("QUPATH")
    config.set("QUPATH", "basepath",        config.get("DEFAULT", "basepath"))
    config.set("QUPATH", "dataset_path",    config.get("DEFAULT", "dataset_path"))
    config.set("QUPATH", "input_type",      config.get("DEFAULT", "input_file_type"))
    config.set("QUPATH", "output_type",     config.get("DEFAULT", "output_file_type"))
    config.set("QUPATH", "qupath_path",     config.get("DEFAULT", "qupath_program_path"))
    config.set("QUPATH", "groovy_script",   config.get("DEFAULT", "qupath_groovy_path"))
    config.set("QUPATH", "parameters_file", config.get("DEFAULT", "qupath_params_path"))

    label_images(window, config)
    config.set("QUPATH", "labeled", "True")
    config.set("DEFAULT", "results_path",     config.get("QUPATH", "results_path"))
    config.set("DEFAULT", "input_file_type",  config.get("QUPATH", "input_type"))
    config.set("DEFAULT", "output_file_type", config.get("QUPATH", "output_type"))

    window.qupath_status.set(f"Labeling & Summary done.")
    window.qupath_progress.set(100)
    config.set("GUI", "btnTrainPathomics_state", "normal")
    window.refresh_buttons()

    # Save configuration under dataset and delete thread.
    with window.mutex, open(config.get("DEFAULT", "configfile"), "w+") as filehandle:
        config.write(filehandle)
    del window.threads["qupath_analyze"]

#############################
# API Functions.            #
#############################
def qupath_params(window):
    params_file = inputs.select_file(msg="Please select QuPath params file.", init_dir=window.assets, types=(("JSON files", "*.json"), ('All files', '*.*')))
    try:
        # If successful, store it as the path of parameters.
        with open(params_file, "r") as filehandle:
            json.load(filehandle)
        config.set("DEFAULT", "qupath_params_path", params)
    except Exception as e:
        pass

def qupath_analyze(window):
    if "qupath_analyze" not in window.threads:
        window.threads["qupath_analyze"] = threading.Thread(target=_qupath_analyze, args=(window,)).start()