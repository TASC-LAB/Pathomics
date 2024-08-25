##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import pickle
import threading

# Third party modules.
import numpy
import boruta
import pandas
import sklearn.ensemble

# Local modules.
import Pathomics.utils

#############################
# API Functions.            #
#############################
def feature_selection(window, config):
    folder = "F:\\University\\Project\\Pathomic 4\\Data"
    label = 'Mouse' #define desired label here

    if label == 'Line':
        rawdf = pandas.concat([rawlabels['Line'],rawdata],axis=1)
        rawdf = rawdf[rawdf.Line != '0000']
        rawlabels = rawlabels[rawlabels.Line != '0000']
        rawdf.drop('Line',axis=1, inplace=True)
        rawdata = rawdf.copy()
        rawdata.to_pickle(os.path.join(folder, 'rawdata_line.pickle'))
        rawlabels.to_pickle(os.path.join(folder, 'rawlabel_line.pickle'))
        # rd = pandas.read_pickle(os.path.join(folder, 'rawdata_line.pickle'))
        # rl = pandas.read_pickle(os.path.join(folder, 'rawdata_line.pickle'))
    # folder = r'\\metlab25\G\AymanData\QuPath\NewPipeline\features' #Path of your pickle files
    # folder = r'\\metlab25\G\AymanData\QuPath\NewPipeline\features\results'
    # data_file = pandas.read_csv(os.path.join(folder, 'QuPathNormalisationFinal.csv')) #Summary table file name

    if label == 'Line':
        rawdata = pandas.read_pickle(os.path.join(folder,'rawdata_line.pickle')) #Change to rawdata_line if running Line
        rawlabel = pandas.read_pickle(os.path.join(folder,'rawlabel_line.pickle'))
    else:
        rawdata = pandas.read_pickle(os.path.join(folder,'rawdata.pickle'))
        rawlabel = pandas.read_pickle(os.path.join(folder,'rawlabel.pickle'))
    # rawdata = pandas.read_pickle(os.path.join(folder,'rawdata_line.pickle')) #Change to rawdata_line if running Line
    rawdatacopy = rawdata.copy()
    zscore_rawdata = Pathomics.utils.zScoreEach(rawdatacopy)
    rawlabel = pandas.read_pickle(os.path.join(folder,'rawlabel.pickle'))
    # rawlabel = pandas.read_pickle(os.path.join(folder,'rawlabel_line.pickle')) #Change to rawlabels_line if running Line

    my_list = zscore_rawdata.columns.values.tolist()
    X = zscore_rawdata[my_list].values
    X = pandas.DataFrame(X).fillna(value=0).values
    y = rawlabel[label].values.ravel()

    if os.path.exists(".\\Data\\boruta.pickle"):
        with open(".\\Data\\boruta.pickle", "rb") as filehandle:
             feat_selector = pickle.load(filehandle)
    else:
        # define random forest classifier, with utilising all cores and
        # sampling in proportion to y labels
        random_forest = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

        # define Boruta feature selection method
        feat_selector = boruta.BorutaPy(random_forest, n_estimators='auto', verbose=2, random_state=1)

        # find all relevant features - 5 features should be selected
        feat_selector.fit(X, y)

        with open(".\\Data\\boruta.pickle", "wb+") as filehandle:
            pickle.dump(feat_selector, filehandle)

    # check selected features - first 5 features are selected
    # feat_selector.support_

    # check ranking of features
    # feat_selector.ranking_
    final_features = list()
    indexes = numpy.where(feat_selector.support_ == True)
    for x in numpy.nditer(indexes):
        final_features.append(my_list[x])

    eminem = rawdata[final_features]
    eminemlabels = rawlabel.copy()
    with open(os.path.join(folder, 'eminem.pickle'), "wb") as f:
        pickle.dump((eminem, eminemlabels), f)