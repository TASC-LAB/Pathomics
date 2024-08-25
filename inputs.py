##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import imageio
import tkinter

# Third party modules.
# import PIL
# from PIL import ImageFont

#############################
# Internal Functions.       #
#############################
# def _get_true_size(text, font_name, font_size):
#     font = ImageFont.truetype(f"{font_name.replace(' ', '').lower()}.ttf", font_size)
#     return font.getlength(text)

#############################
# API Functions.            #
#############################
def count_dataset(path, suffix):
    dataset = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(suffix)]
    dataset = list(map(os.path.abspath,   dataset))
    dataset = list(filter(os.path.isfile, dataset))
    return len(dataset)

def gen_dataset(path, suffix):
    # TODO: Make sure path exists.
    dataset = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(suffix)]
    dataset = list(map(os.path.abspath,   dataset))
    dataset = list(filter(os.path.isfile, dataset))

    for full_path in dataset:

        # Load the dataset in numpy array, 3 dimensions: height, width, channels.
        # The values range from 0 to 255 (Unsigned Byte).
        image = imageio.imread(full_path)

        # TODO: Add support for multiple channels.
        if image.shape[2] == 4:
            image = image[:,:,:3]
        yield full_path, image

def select_file(msg, init_dir=".", types=(('All files', '*.*'))):
    tkinter.messagebox.showinfo("Pathomics 4", msg)
    return os.path.abspath(r"" + tkinter.filedialog.askopenfilename(initialdir=init_dir, filetypes=types))

def select_dir(msg, init_dir="."):
    tkinter.messagebox.showinfo("Pathomics 4", msg)
    return os.path.abspath(r"" + tkinter.filedialog.askdirectory(initialdir=init_dir))

def load_local(window):
    config = window.config

    folder = select_dir("Please select the dataset directory.")
    configfile = os.path.join(folder, config.get("DEFAULT", "configfile"))
    config.read(configfile)
    config.set("DEFAULT", "dataset_path",     folder)
    config.set("DEFAULT", "basepath",         folder)
    config.set("DEFAULT", "configfile",       configfile)
    config.set("DEFAULT", "input_file_type",  window.input_file_type.get())
    config.set("DEFAULT", "output_file_type", window.output_file_type.get())
    config.set("DEFAULT", "width",            window.patch_width.get())
    config.set("DEFAULT", "height",           window.patch_height.get())
    channels = config.get("DEFAULT", "channels").split(',')
    config.set("DEFAULT", "num_of_channels",  str(channels.index(window.patch_channels.get()) + 1))

    window.dataset_path.set("."+os.path.relpath(folder))
    window.pathomics_status.set("Ready for ROI selection.")
    config.set("GUI", "btnROIFilter_state",      "normal")
    config.set("GUI", "btnNormTrainModel_state", "normal")
    config.set("GUI", "btnNormLoadModel_state",  "normal")
    config.set("GUI", "btnQuPathParams_state",   "normal")
    config.set("GUI", "btnQuPathAnalysis_state", "normal")
    # config.set("GUI", "btnTrainPathomics_state", "normal")
    window.refresh_buttons()

def load_mongo(window):
    pass

def load_cloud(window):
    pass