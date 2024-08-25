##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import pickle

# Third party modules.
import numpy
import boruta
import pandas
import IPython
import seaborn
import sklearn
import matplotlib

# Local modules.
import Pathomics.utils

#############################
# Internal Functions.       #
#############################
def _pcaPlot(pca, pca_df, hue, title,nColor=0, nShades=0, nColorTreat=0, nShadesTreat=0, nColorLay=0, nShadesLay=0, xlim_kmeans=[0,0], ylim_kmeans=[0,0]):
    seaborn.axes_style({'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True,'axes.spines.top': True})
    f, ax = matplotlib.pyplot.subplots(figsize=(6.5, 6.5),dpi=100, facecolor='w', edgecolor='k')
    num_of_dep = len(hue.unique())
    seaborn.despine(f, left=True, bottom=True)
    if hue.name=='Experiment' and nColor!=0:
        palette = ChoosePalette(nColor,nShades)
    elif hue.name=='Treatments' and nColorTreat!=0:
        palette = ChoosePalette(nColorTreat,nShadesTreat)
    elif hue.name=='Layers' and nColorLay!=0:
        palette = ChoosePalette(nColorLay,nShadesLay)
    else:
        palette = seaborn.color_palette("hls", num_of_dep)  # Choose color
    pca_expln_var_r = pca.explained_variance_ratio_*100

    s = seaborn.scatterplot(x="PC1", y="PC2", hue=hue.name, data=pca_df, ax=ax, palette=palette);
    matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=1, edgecolor='black')

def _pcaVarianceExplained(pca, NSC):
    f, ax = matplotlib.pyplot.subplots(figsize=(10,10),dpi=100)
    features = range(pca.n_components_);
    matplotlib.pyplot.bar(features, pca.explained_variance_ratio_*100);
    matplotlib.pyplot.xlabel('PCA features')
    matplotlib.pyplot.ylabel('Variance explained %')
    matplotlib.pyplot.xticks(features);
    print('There are ' + '{0:d}'.format(NSC) + ' signficant components')
    # matplotlib.pyplot.show()
    return None

def _pcaCalcOneExp(Groups, data, exp, title, FigureNumber, nColor=0, nShades=0):
    pca = sklearn.decomposition.PCA(random_state=42)
    pca.fit(data)
    pca_transformed = pca.transform(data)
    number_of_significant_components = sum(pca.explained_variance_ratio_>0.1)
    pca_df = pandas.DataFrame(pca_transformed[:,0:2], index=exp.index)
    pca_df.rename(columns={0:'PC1', 1:'PC2'}, inplace=True)

    pca_df[Groups] = [expNames.replace('NNN0','') for expNames in exp]
    IPython.display.display(IPython.display.Latex(r'$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))

    FigureNumber += 1
    _pcaPlot(pca, pca_df, pca_df[Groups], title, nColor, nShades)
    IPython.display.display(IPython.display.Latex(r'$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber += 1
    _pcaVarianceExplained(pca,number_of_significant_components)

    return pca_df, pca, pca_transformed

#############################
# API Functions.            #
#############################
def do_pca(window, config):
    folder = "F:\\University\\Project\\Pathomic 4\\Data"
    with open(os.path.join(folder, 'eminem.pickle'), "rb") as f: #Include the '_line' if you are running Line/QTL analysis!
        rawdata,rawlabels = pickle.load(f)

    features = rawdata.columns.values.tolist()
    Groups = 'Mouse'
    pca_df,pca,pca_transformed = _pcaCalcOneExp(Groups, Pathomics.utils.zScoreEach(rawdata),rawlabels[Groups],'PCA',-1,nColor=0,nShades=0)
    return  pca_df,pca,pca_transformed,Groups