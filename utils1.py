#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, PowerTransformer,Normalizer, MinMaxScaler
from sklearn.mixture import GaussianMixture
import pickle
from IPython.display import display, Latex
from itertools import product, compress
from scipy.stats import *
import math

from statsmodels.formula.api import ols
import statsmodels.api as sm
# 
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


# In[5]:


import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore
from sklearn import decomposition
from IPython.display import display, Latex
from sklearn.cluster import KMeans
import pickle
import os
import pathlib
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import *

# In[6]:
Groups = 'Line'

def ANOVE_DESC_TABLE(dataSpecGraphGroups, Features, title, dep='Groups',groupList=[0,1,2]):
    display(Latex('$\color{blue}{\Large ANOVA\ Table\ feature\ per\ Group}$'))
    ANOVA_MI = pd.MultiIndex.from_product([['Between '+dep,'Within '+dep,'Total'], 
                                           ['Sum of Squares','df','Mean Sqaure','F','Sig.']])
    ANOVA_df = pd.DataFrame(columns=ANOVA_MI,index=Features[:-1])
    index = pd.MultiIndex.from_product([Features[:-1],[0]],
                                       names=['Feature','Sig.'])
    columns = pd.MultiIndex.from_product([[dep],groupList, 
                                           ['N','Mean','Standard Deviation','Standard Deviation Error','95% Upper Bound Mean','95% Lower Bound Mean']])
    ANOVA_Desc_df = pd.DataFrame(columns=columns,index=index)
    ANOVA_Desc_df = ANOVA_Desc_df.reset_index(level='Sig.')
    for par in Features[:-1]:
        model_name = ols(par+' ~ C('+dep+')', data=dataSpecGraphGroups).fit()
        ano = sm.stats.anova_lm(model_name,typ=1)
        ano = ano.append(pd.DataFrame({"df":[ano.df.sum()],"sum_sq":[ano.sum_sq.sum()]},index={"Total"}))
        ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        ano.rename(columns = {"sum_sq": "Sum of Squares", 'mean_sq':'Mean Sqaure', 'PR(>F)':'Sig.'},inplace=True)
        ano.rename(index = {'C('+dep+')': "Between "+dep, 'Residual':'Within '+dep},inplace=True)
        ANOVA_df.at[par] = ano.values.ravel()
        ANOVA_Desc_df.loc[par,'Sig.'] = ANOVA_df.loc[par,'Between '+dep]['Sig.']
        gb_Groups = dataSpecGraphGroups.groupby([dep])[par].agg(['count', 'mean', 'std', ('sem',sem),
                                                                         ('ci95_hi',lambda x:
                                                                                  (np.mean(x) + 1.96*
                                                                                   np.std(x)/math.sqrt(np.size(x)))),
                                                                         ('ci95_lo',lambda x:
                                                                                  (np.mean(x) - 1.96*
                                                                                   np.std(x)/math.sqrt(np.size(x))))])

        ANOVA_Desc_df.loc[par,'Groups'] = gb_Groups.values.ravel()
    display(ANOVA_Desc_df)
    ANOVA_Desc_df.to_csv(title+' ANOVA + Descriptive Table - ' +dep +'.csv')
    return None



# def ANOVA_by_Treatments(dataSpecGraphGroups,Features):
    # for par in Features[:-1]:
        # display(Latex('$\color{blue}{\Large %s}$'%(par)))
        # model_name = ols(par+' ~ C(Groups)', data=dataSpecGraphGroups).fit()
        # ano = sm.stats.anova_lm(model_name,typ=1)
        # ano = ano.append(pd.DataFrame({"df":[ano.df.sum()],"sum_sq":[ano.sum_sq.sum()]},index={"Total"}))
        # ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        # ano.rename(columns = {"sum_sq": "Sum of Squares", 'mean_sq':'Mean Sqaure', 'PR(>F)':'Sig.'},inplace=True)
        # ano.rename(index = {"C(Groups)": "Between Groups", 'Residual':'Within Groups'},inplace=True)
        # display(ano)
    # return None

def ANOVA_TABLE(dataSpecGraphGroups, Features, title='', dep='Groups'):
    display(Latex('$\color{blue}{\Large ANOVA\ Table\ feature\ per\ Group}$'))
    ANOVA_MI = pd.MultiIndex.from_product([['Between Groups','Within Groups','Total'], 
                                           ['Sum of Squares','df','Mean Square','F','Sig.']])
    ANOVA_df = pd.DataFrame(columns=ANOVA_MI,index=Features[:-1])
    for par in Features[:-1]:
        model_name = ols(par+' ~ C('+dep+')', data=dataSpecGraphGroups).fit()
        ano = sm.stats.anova_lm(model_name,typ=1)
        ano = ano.append(pd.DataFrame({"df":[ano.df.sum()],"sum_sq":[ano.sum_sq.sum()]},index={"Total"}))
        ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        ano.rename(columns = {"sum_sq": "Sum of Squares", 'mean_sq':'Mean Square', 'PR(>F)':'Sig.'},inplace=True)
        ano.rename(index = {'C('+dep+')': "Between "+dep, 'Residual':'Within '+dep},inplace=True)
        ANOVA_df.at[par] = ano.values.ravel()
    ANOVA_df.dropna(axis=1,inplace=True)
    ANOVA_df.to_csv(title+' ANOVA Table - ' +dep +'.csv')
    display(ANOVA_df)
    return 

def histogramDataKDE(exp,data,Features,FigureNumber,nColor=0,nShades=0):
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    fig, axes = plt.subplots(nrows=6, ncols=6,figsize=(30,30))
    fig2, ax2 = plt.subplots(figsize=(6,6))
    if nColor==0:
        colors = sns.color_palette("hls", len(exp))
    else:
        colors = ChoosePalette(nColor,nShades)
    for par, ax in zip(Features,axes.flat):
        for label, color in zip(range(len(exp)), colors):
            sns.kdeplot(data[par].loc[data[Groups]==exp[label]], ax=ax, 
                    label=exp[label], color=color, #density=True, stacked=True,
                    )
            ax.set_xlabel(par,fontdict={'fontsize':15}) 
    fig.set_tight_layout(True)
    fig.tight_layout(pad=1.03)
    labels_handles = {label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())}

    fig.set_tight_layout(False)
    for a in axes.flat:
        try:
            a.get_legend().remove()
        except:
            display('No axis a (ignore this message)')
    fig2.legend(labels_handles.values(),
               labels_handles.keys(),
               loc='center',fontsize='small',
               framealpha=1,edgecolor='black'
              )
    # fig.subplots_adjust(top=0.9)

    plt.show()



def TASC(dataDrop, dataLabel, labelsCol='M/NM', LE=True,
        title='All Cell Lines', HC=False, treats=['HGF'], combTreats=[['HGF'],['PHA']],
        LY = 9, TI = 3, multipleCL=True, singleTREAT=False, FigureNumber=2, nrows=0, ncols=1, 
        nColor=0, nShades=0, k_cluster=3,
        nColorTreat=0, nShadesTreat=0, nColorLay=0, nShadesLay=0,
        figsizeEXP=(15,40), figsizeTREATS=(15, 25),figsizeCL=(15, 25), Features='',
        AE_model=True, model_name=''): 
    '''
    Total Analysis - Single Cell:
    Input:
        dataDrop      - data to do analysis on 
        columnsToDrop - which parameters aren't suppose to be as numeric variables
        labelsCol     - 
        LE            - LabelEncoder if the label column isn't a numbers
        title         - title for the main PCA
        HC            - Hirarchical clustering analysis: True/False
    Output:
        
    '''
    dataDrop = raw
    dataAE = dataDrop.copy()
    dataAE[labelsCol] = dataLabel[labelsCol].copy()
    # z-score data
    standardScaler = StandardScaler(with_std=True,)
    dataDrop = pd.DataFrame(standardScaler.fit_transform(dataDrop),columns=dataDrop.columns)
    if HC:
        display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
        FigureNumber+=1
        createHierarchicalCluster(dataDrop,title,'Parameters','# Cell',
                                  sns.diverging_palette(150, 10, n=100),vmin=-2,vmax=2)
    listCells = dataDrop.index.values.copy()
    pca_df,pca,pca_transformed = pcaCalcOneExp(dataDrop, 
                                               dataLabel[labelsCol], 'PCA of '+\
                                               title,FigureNumber, nColor=nColor, nShades=nShades)
    FigureNumber+=2
    ## Find the Best K for k-means
    # display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    # FigureNumber+=1
    # ElbowGraph(pca,pca_transformed)
    ## K-means
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    kmeans_pca,xlim_kmeans,ylim_kmeans = kmeansPlot(k_cluster,pca_transformed,pca,dataLabel['Experiment'])   
    ## Times (Time point)
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    pca_df['Time'] = dataLabel['TimeIndex'].copy()
    pcaPlot(pca, pca_df, pca_df['Time'], 'by Time points', nColor, nShades)
    

    
def pcaCalcOneExp(data,exp,title,FigureNumber,nColor=0,nShades=0):
    pca = decomposition.PCA(random_state=42)
    pca.fit(data)
    pca_transformed = pca.transform(data)
    number_of_significant_components = sum(pca.explained_variance_ratio_>0.1)
    pca_df = pd.DataFrame(pca_transformed[:,0:2],index=exp.index)

    pca_df.rename(columns={0:'PC1', 1:'PC2'}, inplace=True)
    
    pca_df[Groups] = [expNames.replace('NNN0','') for expNames in exp]
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))

    FigureNumber+=1    
    pcaPlot(pca, pca_df, pca_df[Groups], title, nColor, nShades)  
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    FigureNumber+=1
    pcaVarianceExplained(pca,number_of_significant_components)
    
    return pca_df, pca, pca_transformed


def pcaVarianceExplained(pca,NSC):
    f, ax = plt.subplots(figsize=(10,10),dpi=100) 
    features = range(pca.n_components_);
    plt.bar(features, pca.explained_variance_ratio_*100);
    plt.xlabel('PCA features')
    plt.ylabel('Variance explained %')
    plt.xticks(features);
    print('There are ' + '{0:d}'.format(NSC) + ' signficant components')
    plt.show()
    return None

def pcaPlot(pca, pca_df, hue, title,nColor=0, nShades=0, nColorTreat=0, nShadesTreat=0,
            nColorLay=0,nShadesLay=0,xlim_kmeans=[0,0],ylim_kmeans=[0,0]):
    sns.axes_style({'axes.spines.left': True, 'axes.spines.bottom': True, 
               'axes.spines.right': True,'axes.spines.top': True})
    f, ax = plt.subplots(figsize=(6.5, 6.5),dpi=100, facecolor='w', edgecolor='k')
    num_of_dep = len(hue.unique())
    sns.despine(f, left=True, bottom=True)
    if hue.name=='Experiment' and nColor!=0:
        palette = ChoosePalette(nColor,nShades)
    elif hue.name=='Treatments' and nColorTreat!=0:
        palette = ChoosePalette(nColorTreat,nShadesTreat)
    elif hue.name=='Layers' and nColorLay!=0:
        palette = ChoosePalette(nColorLay,nShadesLay)
    else:
        palette = sns.color_palette("hls", num_of_dep)  # Choose color  
    pca_expln_var_r = pca.explained_variance_ratio_*100
    
    s = sns.scatterplot(x="PC1", y="PC2", hue=hue.name, data=pca_df, ax=ax, 
                        palette=palette);

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,framealpha=1,edgecolor='black')
    
    

def kmeansPlot(k_cluster,pca_transformed,pca,dataLabel):
    number_of_significant_components = sum(pca.explained_variance_ratio_>=0.1)
    if number_of_significant_components<2:
        number_of_significant_components = 2
	
    pca_transformed_n = pca_transformed[:,0:number_of_significant_components]
    f, ax = plt.subplots(figsize=(6.5, 6.5),dpi=100, facecolor='w', edgecolor='k')
    pca_expln_var_r = pca.explained_variance_ratio_*100
    PC_col = ['PC'+str(x) for x in range(1,number_of_significant_components+1)]
    kmeans_pca = pd.DataFrame(pca_transformed_n, columns=PC_col, index=dataLabel.index)
    kmeanModel = KMeans(n_clusters=k_cluster, n_jobs=-1, random_state=0).fit(pca_transformed_n)
    #kmeanModel = KMeans(n_clusters=k_cluster, init ='k-means++', n_init = 50, n_jobs=-1, random_state=0).fit(pca_transformed_n)
    idx = np.argsort(kmeanModel.cluster_centers_.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(k_cluster)
    kmeans_pca['Groups'] = lut[kmeanModel.predict(pca_transformed_n)]
    num_of_dep = len(kmeans_pca['Groups'].unique())
    sns.despine(f, left=True, bottom=True)
    palette = sns.color_palette("hls", num_of_dep)  # Choose color  
    s = sns.scatterplot(x="PC1", y="PC2", hue = 'Groups', data=kmeans_pca, ax=ax,
                        legend='full', palette=palette);
    plt.suptitle('K-means clustering k=' + '{0:.0f}'.format(k_cluster), fontdict={'fontweight':'bold', 'fontsize':25})
    plt.xlabel('PC1 (' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%)', fontdict={'fontsize':15});
    plt.ylabel('PC2 (' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%)', fontdict={'fontsize':15});

    ## splitting the legend list into few columns
    if len(ax.get_legend().texts)>25:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=3,framealpha=1,edgecolor='black')
    elif len(ax.get_legend().texts)>17:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2,framealpha=1,edgecolor='black')
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1,framealpha=1,edgecolor='black')
    xlim_kmeans_l,xlim_kmeans_r = plt.xlim()
    ylim_kmeans_l,ylim_kmeans_r = plt.ylim()
    xlim_kmeans = [xlim_kmeans_l,xlim_kmeans_r]
    ylim_kmeans = [ylim_kmeans_l,ylim_kmeans_r]
    centers = kmeanModel.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=25, );
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.show()
    return kmeans_pca,xlim_kmeans,ylim_kmeans


def pcaCalc(data,expNamesInOrder,title,nColor=1,nShades=2):
    pca = decomposition.PCA(random_state=42)
    pca.fit(data)
    pca_transformed = pca.transform(data)
    number_of_significant_components = sum(pca.explained_variance_ratio_>0.1)
    pca_df = pd.DataFrame(pca_transformed[:,0:2],index=expNamesInOrder.index.values)

    pca_df.rename(columns={0:'PC1', 1:'PC2'}, inplace=True)

    pca_df[Groups] = [expNames.replace('NNN0','') for expNames in expNamesInOrder]
    
    pcaPlot(pca, pca_df, pca_df[Groups], title,nColor,nShades)
    return None


def histByKmeansTreatsLabel(pca_df,Par,k_cluster=3,bar_width=0.2,figsize=(10,5),labels='',rotate=0):
    fig, ax = plt.subplots(figsize=(15,5),dpi=100)
    Treat = pca_df.groupby(Par)
    print(Treat)
    rangeLabel = [float(i) for i in list(Treat.describe().index.values)]
    rangeLabelX = [float(i)+bar_width for i in list(Treat.describe().index.values)]
    for j in list(Treat.describe().index.values):
        T = Treat.get_group(j)['Groups']
        xlabels = T.unique()
        xlabels.sort()
        N = len(xlabels)
        color = sns.color_palette('hls',k_cluster)
        xrange = range(N)
        SUM = T.value_counts().sort_index().sum()
        for i in range(k_cluster):
            Group = T.loc[T==i].value_counts().sort_index()
            if len(xlabels)==k_cluster:
                plt.bar(rangeLabel[j] + i*bar_width, Group/SUM, bar_width,
                        label='Group '+str(i),
                        color=color[i])
            else:
                for e in xlabels:
                    if e not in Group:
                        Group[e] = 0
                Group.sort_index()
                plt.bar(rangeLabel[j] + i*bar_width, Group/SUM, bar_width, 
                        label='Group '+str(i),
                        color=color[i])
    if labels=='':
        plt.xticks(xrange + bar_width, (xlabels), rotation=rotate, fontsize=12)
    else:
        labels.sort()
        plt.xticks(rangeLabelX , (labels), rotation=rotate, fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),bbox_to_anchor=(1.15,1),framealpha=1,edgecolor='black')
    plt.xlabel(Par,fontdict={'fontsize':15})
    plt.suptitle('each '+Par+'is 100%',fontdict={'fontsize':15}) 
    
    plt.show()
    return None


def histByKmeans(pca_df,Par,k_cluster=3,bar_width=0.2,figsize=(10,5),labels='',rotate=0):
    fig, ax = plt.subplots(figsize=figsize,dpi=100)
    plt.title(Par+' Histogram by group',fontdict={'fontsize':20})
    xlabels = pca_df[Par].unique()
    xlabels.sort()
    xrange = np.arange(0,len(xlabels),1)
    for i in range(k_cluster):
        Group = pca_df[Par].loc[pca_df['Groups']==i].value_counts().sort_index()
        plt.bar(xlabels + i*bar_width, Group/Group.sum(), bar_width, label='Group '+str(i))
        
    plt.legend(bbox_to_anchor=(1.2,1),framealpha=1,edgecolor='black')
    if labels=='':
        plt.xticks(xlabels + bar_width, (xlabels),rotation=rotate, fontsize=12)
    else:
        plt.xticks(xlabels + bar_width, (labels),rotation=rotate, fontsize=12)
    plt.xlabel(Par + ' range',fontdict={'fontsize':15}) 
    plt.show()
    return None


# In[7]:


def PCA_colorPar1_titlePar2(pca, pca_df, Par1, Par2, figsize, labelsPar1='', labelsPar2='',
                            nrows=0, nColor=0, nShades=0, nColorTreat=0,nShadesTreat=0,
                            nColorLay=0, nShadesLay=0,xlim_kmeans=[-10,10],ylim_kmeans=[-10,10]):

    fig2,ax2 = plt.subplots(figsize=(10,6), facecolor='w', edgecolor='k')
    title = 'Each graph is a seperate ' + Par2 + ' and colored by ' + Par1 
    num_of_dep = len(pca_df[Par1].unique())
    if Par1=='Experiment' and nColor!=0:
        palette = ChoosePalette(nColor,nShades)
    elif Par1=='Treatments' and nColorTreat!=0:
        palette = ChoosePalette(nColorTreat,nShadesTreat)
    elif Par1=='Layers' and nColorLay!=0:
        palette = ChoosePalette(nColorLay,nShadesLay)
    else:
        palette = sns.color_palette("hls", num_of_dep)  # Choose color  
    pca_expln_var_r = pca.explained_variance_ratio_*100
    uPar2 = pca_df[Par2].unique()
    uPar2.sort()
    uPar1 = list(pca_df[Par1].unique())
    uPar1.sort()
    if nrows==0:
        nrows = int(np.floor(np.sqrt(len(uPar2))))
        if nrows<=1:
            nrows = 1
            ncols = len(uPar2)
        elif nrows**2==len(uPar2):
            nrows = nrows
            ncols = nrows
        else:
            if len(uPar2)%nrows==0:
                # nrows += 1
                ncols = int(len(uPar2)/nrows)
            else:
                ncols = nrows + 1
                nrows += 1
    else:
        ncols = int(len(uPar2)/nrows)
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, 
                           ncols=ncols, constrained_layout=True,dpi=100)
    for i, a, t in zip(uPar2,ax.reshape(-1), range(len(uPar2))):
        df_PCA = pca_df.loc[pca_df[Par2]==i].copy()
        uPar1tmp = df_PCA[Par1].unique()
        uPar1tmp.sort()
        pal = sns.color_palette([palette[p] for p in [uPar1.index(value) for value in uPar1tmp]])
        sns.despine(fig, left=True, bottom=True)
        s = sns.scatterplot(x="PC1", y="PC2", hue=Par1, data=df_PCA, ax=a, 
                            palette=pal);
        if labelsPar2=='':
            a.set_title(i, fontweight='bold', fontsize=15);
        else:
            a.set_title(labelsPar2[t], fontweight='bold', fontsize=15);
        a.set_xlabel('PC1 ' + '{0:.2f}'.format(pca_expln_var_r[0]) + '%');
        a.set_ylabel('PC2 ' + '{0:.2f}'.format(pca_expln_var_r[1]) + '%');
        a.set_xlim(xlim_kmeans)
        a.set_ylim(ylim_kmeans)
        # a.set_xlim([pca_df['PC1'].min()+0.1*pca_df['PC1'].min(), pca_df['PC1'].max()+0.1*pca_df['PC1'].max()])
        # a.set_ylim([pca_df['PC2'].min()+0.1*pca_df['PC2'].min(), pca_df['PC2'].max()+0.1*pca_df['PC2'].max()])
        
    if labelsPar1!='':
        labels_handles = {
          lPar1: handle for ax in fig.axes for handle, label, lPar1 in zip(*ax.get_legend_handles_labels(),labelsPar1)
        }
    else:
        labels_handles = {
          label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
        }
    # fig.set_tight_layout(True)

    # fig.tight_layout(pad=1.02)

    # fig.legend(
                # labels_handles.values(),
                # labels_handles.keys(),
                # loc="upper center",
                # bbox_to_anchor=(1.2, 1),
                # bbox_transform=plt.gcf().transFigure,
                # )
    for a in ax.flat:
        try:
            a.get_legend().remove()
            for spine in a.spines.values():
                spine.set_visible(True)
        except:
            print('')
        
    # for item in [fig, ax]:
    # fig.patch.set_visible(True)
    fig2.legend(labels_handles.values(),
           labels_handles.keys(),
           loc='center',fontsize='xx-large',
           framealpha=1,edgecolor='black',
          )
    fig2.subplots_adjust(right=0.5,top=0.9)
    # fig.set_tight_layout(False)
    plt.suptitle(title, fontweight='bold', fontsize=25)

    plt.show()
    return None


# In[9]:


def histogramDataKDELabels(Labels,data,Features,FigureNumber,Par='Experiment',nColor=0,nShades=0):
    display(Latex('$\color{blue}{\Large Figure\ %i}$'%(FigureNumber)))
    fig, axes = plt.subplots(nrows=6, ncols=6,figsize=(30,30),dpi=100)
    fig2,ax2 = plt.subplots(figsize=(6,6))
    if nColor==0:
        colors = sns.color_palette("hls", len(Labels))
    else:
        colors = ChoosePalette(nColor,nShades)
    # 
    for par, ax in zip(Features,axes.flat):
        for label, color in zip(range(len(Labels)), colors):
            sns.kdeplot(data[par].loc[data[Par]==Labels[label]], ax=ax, 
                    label=Labels[label], color=color, bw=0.7
                    )
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
    # plt.subplots_adjust(right=0.8)
    plt.show()
    return None


# In[10]:


def zScoreEach(data):
    data = pd.DataFrame(data)
    for col in data.columns:
        data[col] = zscore(data[col]).astype(float)
    return data


# In[11]:

#Function dealing with column headings
def to_groups(df):
    df["Groups"]=0
    headings= df.columns.tolist()
    for t in range(len(headings)):
        if headings[t]=='0':
            headings[t]='Group A'
        if headings[t]=='1':
            headings[t]='Group B'
        if headings[t]=='2':
            headings[t]='Group C'
        if headings[t]=='3':
            headings[t]='Group D'
        if headings[t]=='4': 
            headings[t]='Group E'
        if headings[t]=='5':     
            headings[t]='Group F'
        if headings[t]=='6':    
            headings[t]='Group G'
        t+=1
    df.columns=headings

    for i in range(len(data2)):
        if df["Group A"][i]==1:
            df["Groups"][i]= 'A'
        if df["Group B"][i]==1:
            df["Groups"][i]= 'B'
        if df["Group C"][i]==1:
            df["Groups"][i]= 'C'
        i+=1
    return(df)


