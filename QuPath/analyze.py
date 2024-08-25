##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import re
import shutil
import logging
import threading
import subprocess

# Third party modules.
import pandas

# Local modules.
import inputs

##############################
# CONSTANTS                  #
##############################
QUPATH_IMAGE_OUT_TYPE = ".tsv"

#############################
# Internal Functions.       #
#############################
def _setup_qupath(window, config):
    basepath = r"" + os.path.join(config.get("DEFAULT", "basepath"), "QuPath")

    config.set("QUPATH", "processed_path", os.path.join(basepath, "Processed"))
    config.set("QUPATH", "results_path",   os.path.join(basepath, "Results"))
    os.makedirs(config.get("QUPATH", "processed_path"), exist_ok=True)
    os.makedirs(config.get("QUPATH", "results_path"),   exist_ok=True)

    # Read QuPath parameters.
    parameters = ""
    with open(config.get("QUPATH", "parameters_file")) as fs:
        for line in fs:
            parameters+=line.strip()
    parameters = parameters.replace(" ", "____")
    config.set("QUPATH", "parameters", parameters)

def _create_summary_table(window, config):
    output_dir   = config.get("QUPATH", "processed_path")
    results_path = config.get("QUPATH", "results_path")

    average_all = pandas.DataFrame([])
    tsv_files   = list(filter(lambda file: file.endswith(QUPATH_IMAGE_OUT_TYPE), os.listdir(os.path.join(output_dir, "QPProject"))))
    for file in tsv_files:
        print(f"Processing image: {file}")
        df = pandas.read_csv(os.path.join(output_dir, "QPProject", file), sep = '\t')

        # transform headers to original format
        df.columns = [re.sub(r' ', '', col) for col in df.columns]
        df.columns = [re.sub(r'[\(\)\:\.]', '_', col) for col in df.columns]

        image   = df.iloc[0,0]
        numbers = df.iloc[:,7:]
        average = numbers.mean(axis = 0, skipna = False)
        df_average = pandas.DataFrame([average])
        df_average.insert(0,'Image', image)
        average_all = pandas.concat([average_all, df_average])
        print(f"Finished processing image: {file}")
    summary_table_path = os.path.join(results_path, config.get("DEFAULT", "summary_table_file"))
    average_all.to_csv(summary_table_path)

    logging.info(f"QuPath image extraction completed successfully. Output directory: {results_path}")
    logging.info(f"Summary table saved to: {summary_table_path}")

def _image_extract(window, config):
    input_dir     = config.get("QUPATH", "dataset_path")
    output_dir    = config.get("QUPATH", "processed_path")
    input_type    = config.get("QUPATH", "input_type")
    groovy_script = config.get("QUPATH", "groovy_script")
    qupath_path   = config.get("QUPATH", "qupath_path")
    parameters    = config.get("QUPATH", "parameters")

    # Set up logging
    log_path    = os.path.join(output_dir, "qupath_image_extraction.log")
    log_format  = "%(asctime)s %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    config.set("QUPATH", "log_path",   log_path)
    config.set("QUPATH", "log_format", log_format)
    logging.basicConfig(filename=log_path, level=logging.INFO, format=log_format, datefmt=date_format)
    logging.info(f"Starting QuPath image extraction with the following parameters: image_dir={input_dir}, output_dir={output_dir}, image_type={input_type}, groovy_script={groovy_script}, program_path={qupath_path}")

    # Execute QuPath groovy script.
    num_of_images = inputs.count_dataset(input_dir, input_type)
    dataset       = inputs.gen_dataset(input_dir, input_type)
    temp_dir      = os.path.join("input_dir", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    for image_index, (path, image) in enumerate(dataset):
        temp_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.copyfile(path, temp_path)
        result = subprocess.run([qupath_path, "script", groovy_script, "-a", temp_dir,"-a", output_dir, "-a", parameters, "-a", input_type], capture_output=True)
        logging.info(f"Image #{image_index + 1} - QuPath output:\n{result.stdout.decode('utf-8')}")
        os.remove(temp_path)
        window.qupath_progress.set(100 * (image_index + 1) / num_of_images)
    os.rmdir(temp_dir)
    _create_summary_table(window, config)

#############################
# API Functions.            #
#############################
def label_images(window, config):
    path        = config.get("QUPATH", "dataset_path")
    input_type  = config.get("QUPATH", "input_type")
    dataset     = inputs.gen_dataset(path, input_type)

    _setup_qupath(window, config)
    _image_extract(window, config)