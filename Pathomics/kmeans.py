##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import pickle

# Third party modules.
import numpy
import pandas
import IPython
import seaborn
import sklearn
# import sklearn.cluster
import matplotlib

#############################
# Internal Functions.       #
#############################
def _kmeansPlot(k_cluster, pca_transformed, pca, dataLabel):
    number_of_significant_components = sum(pca.explained_variance_ratio_>=0.1)
    if number_of_significant_components<2:
        number_of_significant_components = 2

    pca_transformed_n = pca_transformed[:,0:number_of_significant_components]
    f, ax = matplotlib.pyplot.subplots(figsize=(6.5, 6.5),dpi=100, facecolor='w', edgecolor='k')
    pca_expln_var_r = pca.explained_variance_ratio_*100
    PC_col = ['PC'+str(x) for x in range(1,number_of_significant_components+1)]
    kmeans_pca = pandas.DataFrame(pca_transformed_n, columns=PC_col, index=dataLabel.index)
    kmeanModel = sklearn.cluster.KMeans(n_clusters=k_cluster, random_state=0).fit(pca_transformed_n)
    # kmeanModel = sklearn.cluster.KMeans(n_clusters=k_cluster, n_jobs=-1, random_state=0).fit(pca_transformed_n)
    #kmeanModel = sklearn.cluster.KMeans(n_clusters=k_cluster, init ='k-means++', n_init = 50, n_jobs=-1, random_state=0).fit(pca_transformed_n)
    idx = numpy.argsort(kmeanModel.cluster_centers_.sum(axis=1))
    lut = numpy.zeros_like(idx)
    lut[idx] = numpy.arange(k_cluster)
    kmeans_pca['Groups'] = lut[kmeanModel.predict(pca_transformed_n)]
    num_of_dep = len(kmeans_pca['Groups'].unique())
    seaborn.despine(f, left=True, bottom=True)
    palette = seaborn.color_palette("hls", num_of_dep)  # Choose color
    s = seaborn.scatterplot(x="PC1", y="PC2", hue = 'Groups', data=kmeans_pca, ax=ax, legend='full', palette=palette);
    matplotlib.pyplot.suptitle('K-means clustering k=' + '{0:.0f}'.format(k_cluster), fontdict={'weight':'bold', 'size':25})
    # matplotlib.pyplot.suptitle('K-means clustering k=' + '{0:.0f}'.format(k_cluster), fontdict={'fontweight':'bold', 'fontsize':25})
    matplotlib.pyplot.xlabel('PC1 (' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%)', fontdict={'fontsize':15});
    matplotlib.pyplot.ylabel('PC2 (' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%)', fontdict={'fontsize':15});

    ## splitting the legend list into few columns
    if len(ax.get_legend().texts)>25:
        matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=3,framealpha=1,edgecolor='black')
    elif len(ax.get_legend().texts)>17:
        matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2,framealpha=1,edgecolor='black')
    else:
        matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1,framealpha=1,edgecolor='black')
    xlim_kmeans_l,xlim_kmeans_r = matplotlib.pyplot.xlim()
    ylim_kmeans_l,ylim_kmeans_r = matplotlib.pyplot.ylim()
    xlim_kmeans = [xlim_kmeans_l,xlim_kmeans_r]
    ylim_kmeans = [ylim_kmeans_l,ylim_kmeans_r]
    centers = kmeanModel.cluster_centers_
    matplotlib.pyplot.scatter(centers[:, 0], centers[:, 1], c='black', s=25, );
    for spine in ax.spines.values():
        spine.set_visible(True)
    # matplotlib.pyplot.show()
    return kmeans_pca,xlim_kmeans,ylim_kmeans

def _histogramDataKDELabels(Labels,data,Features,FigureNumber,Par='Experiment',nColor=0,nShades=0):
    IPython.display.display(IPython.display.Latex(r'$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    fig, axes = matplotlib.pyplot.subplots(nrows=6, ncols=6,figsize=(30,30),dpi=100)
    fig2,ax2 = matplotlib.pyplot.subplots(figsize=(6,6))
    if nColor==0:
        colors = seaborn.color_palette("hls", len(Labels))
    else:
        colors = ChoosePalette(nColor,nShades)

    for par, ax in zip(Features,axes.flat):
        for label, color in zip(range(len(Labels)), colors):
            seaborn.kdeplot(data[par].loc[data[Par]==Labels[label]], ax=ax, label=Labels[label], color=color, bw_method=0.7)
            ax.set_xlabel(par,)
    fig.set_tight_layout(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.set_tight_layout(False)
    for a in axes.flat:
        try:
            a.get_legend().remove()
        except:
            print('')
    # fig.tight_layout(pad=1.05)
    fig2.legend(handles, labels, loc='upper right',fontsize='xx-large',framealpha=1,edgecolor='black')
    # matplotlib.pyplot.subplots_adjust(right=0.8)

    return fig2,fig

#############################
# API Functions.            #
#############################
def do_kmeans(window, config, res_pca):
    folder = "F:\\University\\Project\\Pathomic 4\\Data"
    pca_df,pca,pca_transformed,Groups = res_pca
    ###################
    # Imported here to facilitate file separation.
    with open(os.path.join(folder, 'eminem.pickle'), "rb") as f: #Include the '_line' if you are running Line/QTL analysis!
        rawdata,rawlabels = pickle.load(f)
    ###################

    K=range(1,12)
    wss = []
    for k in K:
        kmeans=sklearn.cluster.KMeans(n_clusters=k,init="k-means++")
        kmeans=kmeans.fit(rawdata)
        wss_iter = kmeans.inertia_
        wss.append(wss_iter)

    #Elbow diagram
    mycenters = pandas.DataFrame({'Clusters' : K, 'WSS' : wss})
    seaborn.scatterplot(x = 'Clusters', y = 'WSS', data = mycenters, marker="+")

    #Silhouette extraction optimizing number of clusters
    max=0
    clusters=0
    for i in range(3,13):
        labels=sklearn.cluster.KMeans(n_clusters=i,init="k-means++",random_state=200).fit(rawdata).labels_
        if max < sklearn.metrics.silhouette_score(rawdata,labels,metric="euclidean",sample_size=1000,random_state=200):
                max= sklearn.metrics.silhouette_score(rawdata,labels,metric="euclidean",sample_size=1000,random_state=200)
                num_of_clusters=i

    kmeans_pca,xlim_kmeans,ylim_kmeans = _kmeansPlot(num_of_clusters,pca_transformed,pca,rawlabels[Groups])
    return kmeans_pca,xlim_kmeans,ylim_kmeans,Groups