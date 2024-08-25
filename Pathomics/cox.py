##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import math
import pickle

# Third party modules.
import numpy
import pandas
import IPython
import seaborn
import sklearn
import sklearn.cluster
import matplotlib

#############################
# Internal Functions.       #
#############################

#############################
# API Functions.            #
#############################
def do_cox(window, config, res_kmeans):
    folder = "F:\\University\\Project\\Pathomic 4\\Data"
    kmeans_pca,xlim_kmeans,ylim_kmeans,Groups = res_kmeans
    ###################
    # Imported here to facilitate file separation.
    with open(os.path.join(folder, 'eminem.pickle'), "rb") as f: #Include the '_line' if you are running Line/QTL analysis!
        rawdata,rawlabels = pickle.load(f)
    ###################

    array = rawlabels[Groups]
    labels = list(array)
    kgroups = kmeans_pca['Groups']
    pred = list(kgroups)

    # Create a DataFrame with labels and varieties as columns: df
    df = pandas.DataFrame({'Labels': labels, 'Clusters': pred})

    # Create crosstab: ct
    ct = pandas.crosstab(df['Labels'], df['Clusters'],normalize = 'index')

    # Display ct
    fig, ax = matplotlib.pyplot.subplots(figsize=(20,10))
    seaborn.heatmap(ct, annot=True)
    ax.set_ylim([len(numpy.unique(array)),0])
    for t in ax.texts: t.set_text(math.floor(float(t.get_text())*100))
    for t in ax.texts: t.set_text((t.get_text()) + "%")
    # matplotlib.pyplot.show()