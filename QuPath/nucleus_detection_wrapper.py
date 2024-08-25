import os
import subprocess
import pandas as pd
import argparse
import re
import logging

def run_qupath_image_extraction(image_dir, output_dir=None, image_type='svs', groovy_script="qupath_script.groovy" ,config_file='qupath_parameters.json', program_path="QuPath-0.5.1 (console).exe"):
    if output_dir is None:
        output_dir = os.path.join(image_dir, "qupath_output/")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Set up logging
    logging.basicConfig(filename=os.path.join(output_dir, 'qupath_image_extraction.log'), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f"Starting QuPath image extraction with the following parameters: image_dir={image_dir}, output_dir={output_dir}, image_type={image_type}, groovy_script={groovy_script}, config_file={config_file}, program_path={program_path}")

    qupath_parameters = ""
    with open(config_file) as fs:
        for line in fs:
            qupath_parameters+=line.strip()
    qupath_parameters = qupath_parameters.replace(" ", "____")
    result = subprocess.run([program_path, "script", groovy_script, "-a", image_dir,"-a", output_dir, "-a", qupath_parameters, "-a", image_type], capture_output=True)
    logging.info(f"QuPath subprocess output:\n{result.stdout.decode('utf-8')}")
    # for line in result.stdout.split(b'\r\n'):
    #     print(str(line))

    average_all = pd.DataFrame([])
    for file in os.listdir(os.path.join(output_dir, "QPProject")):
        if file.endswith('.tsv'):
            print(f"Processing image: {file}")
            df = pd.read_csv(os.path.join(output_dir, "QPProject", file), sep = '\t')
            # transform headers to original format
            df.columns = [re.sub(r' ', '', col) for col in df.columns]
            df.columns = [re.sub(r'[\(\)\:\.]', '_', col) for col in df.columns]

            image = df.iloc[0,0]
            numbers = df.iloc[:,7:]
            average = numbers.mean(axis = 0, skipna = False)
            df_average = pd. DataFrame([average])
            df_average.insert(0,'Image', image)
            average_all = pd.concat([average_all, df_average])
            print(f"Finished processing image: {file}")

    summary_table_path = os.path.join(output_dir, "Summary_Table.csv")
    average_all.to_csv(summary_table_path)

    logging.info(f"QuPath image extraction completed successfully. Output directory: {output_dir}")
    logging.info(f"Summary table saved to: {summary_table_path}")
    print(f"QuPath image extraction completed successfully. Output directory: {output_dir}")
    print(f"Summary table saved to: {summary_table_path}")

    
def main():
    args = parser.parse_args()
    print("Starting QuPath image extraction...")
    run_qupath_image_extraction(args.input_image_dir, args.outdir, args.image_type, args.nucleus_detection_automation, args.qupath_parameters_config, args.qupath_console_path)
    print("QuPath image extraction completed.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=".", help="Path to ouput directory")
    parser.add_argument("--qupath_console_path", type=str, default=r"/home/ilants/Documents/QuPath-v0.5.1-Linux/QuPath/bin/QuPath", help="Path to the qupath console application")
    parser.add_argument("--input_image_dir", type=str, default=r"/home/ilants/Documents/example_image_dir", help="path to directory containing all images (one format of images for example .svs)")
    parser.add_argument("--image_type", type=str, default="svs", help="image type (for example .svs|.png|.tiff)")
    parser.add_argument("--nucleus_detection_automation", type=str, default=r"/home/ilants/Documents/utls/nucleus_detection_wargs.groovy", help="path to the groovy script that preforms the nucleus detection automation in qupath")
    parser.add_argument("--qupath_parameters_config", type=str, default=r"/home/ilants/Documents/utls/qupath_parameters.json", help="path to the parameters json file")

    main()