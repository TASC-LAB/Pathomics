##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import threading

# Third party modules.
import pandas
import matplotlib
import matplotlib.pyplot

# Local modules.
from .boruta import feature_selection
from .pca    import do_pca
from .kmeans import do_kmeans
from .cox    import do_cox
# from .anova  import anova

#############################
# Internal Functions.       #
#############################
def _read_summary_table(window, config):
    folder = "F:\\University\\Project\\Pathomic 4\\Data"
    data_file = pandas.read_csv(os.path.join(folder, config.get("DEFAULT", "summary_table_file")))
    rawdata = data_file.iloc[:,1:]
    rawdata.to_pickle(os.path.join(folder, 'rawdata.pickle'))
    labels = data_file = data_file.iloc[:,:1]
    mousedata = {'Mouse':data_file['Image'].str[2:6]} #Gets the mouse characters from the 32 character code in a seperate vector
    linedata = {'Line':data_file['Image'].str[21:25]}
    slicedata = {'Series':data_file['Image'].str[38:47]}
    posdata = {'Patch Position (X_Y)':data_file['Image'].str[48:]}
    codedata = {'CodeSeries':data_file['Image'].str[:47]}
    Mouse = pandas.DataFrame(mousedata)
    Lines = pandas.DataFrame(linedata)
    Slice = pandas.DataFrame(slicedata)
    Position = pandas.DataFrame(posdata)
    Code = pandas.DataFrame(codedata)
    Position['Patch Position (X_Y)'] = Position['Patch Position (X_Y)'].str.replace('.png.png.txt', '')
    labels = pandas.concat([Mouse,Lines,Slice,Code,data_file['Image']],axis = 1)
    X_ROI = []
    Y_ROI = []

    for i in Position['Patch Position (X_Y)']:
        d= i.split('_',1)
        xroi = int(d[0])
        yroi = int(d[1])
        X_ROI.append(xroi)
        Y_ROI.append(yroi)

    labels = labels.reset_index()
    #sumTable = sumTable.reset_index()
    labels = pandas.concat([labels,pandas.DataFrame(X_ROI, columns=["X_ROI"]), pandas.DataFrame(Y_ROI, columns=["Y_ROI"])],axis = 1)
    # inem = inem.reset_index()
    # inem2 = pandas.concat([labels,inem],axis = 1)
    del labels['index']
    #linelabels = linelabels.sort_values(by = label , ascending=False) #Sorting the data by Line (top to bottom)

    labels.to_pickle(os.path.join(folder, 'rawlabel.pickle'))
    rawlabels = pandas.read_pickle(os.path.join(folder, 'rawlabel.pickle'))

def _pathomics_train(window):
    config = window.config

    matplotlib.pyplot.switch_backend('Agg')
    dataset = _read_summary_table(window, config)
    feature_selection(window, config)
    res_pca = do_pca(window, config)
    res_kmeans = do_kmeans(window, config, res_pca)
    do_cox(window, config, res_kmeans)
    # kaplan_meier(window, config)
    # qtl(window, config)
    # results(window, config) # log rank test
    # feature_distributions(window, config)
    # anova(window, config)
    # mark_groups(window, config)
    matplotlib.pyplot.switch_backend('TkAgg')
    print("DONE")

#############################
# API Functions.            #
#############################
def pathomics_train(window):
    if "pathomics_train" not in window.threads:
        window.threads["pathomics_train"] = threading.Thread(target=_pathomics_train, args=(window,)).start()

def pca_show(window):
    pass

def kmeans_show(window):
    pass

def cox_analysis_show(window):
    pass

def features_show(window):
    pass

def anova_show(window):
    pass

def kaplan_meier_show(window):
    pass

def qtl_show(window):
    pass

def generate_report(window):
    pass