##############################
# Package Imports.           #
##############################
# Third party modules.
import numpy
import scipy
import pandas

##############################
# SETUP                      #
##############################
setattr(numpy, 'int',   numpy.int32)
setattr(numpy, 'float', numpy.float64)
setattr(numpy, 'bool',  numpy.bool_)

#############################
# API Functions.            #
#############################
def zScoreEach(data):
    data = pandas.DataFrame(data)
    for col in data.columns:
        data[col] = scipy.stats.zscore(data[col]).astype(float)
    return data

# Function dealing with column headings
def to_groups(df):
    df=pandas.DataFrame(df)
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

    for i in range(len(df)):
        if df["Group A"][i]==1:
            df["Groups"][i]= 'A'
        if df["Group B"][i]==1:
            df["Groups"][i]= 'B'
        if df["Group C"][i]==1:
            df["Groups"][i]= 'C'
        i+=1
    return(df)

def cox(data_cox):
    for i in range(len(data_cox)):
        if data_cox['Groups'][i]== 'A':
            data_cox['Groups'][i]= 1
        if data_cox['Groups'][i]== 'B':
            data_cox['Groups'][i]= 2
        if data_cox['Groups'][i]== 'C':
            data_cox['Groups'][i]= 3
        if data_cox['Groups'][i]== 'Ignore':
            data_cox['Groups'][i]= 0
        #if data_cox['Groups'][i]== 'D':
            #data_cox['Groups'][i]= 4
        #if data_cox['Groups'][i]== 'E':
            #data_cox['Groups'][i]= 5
        #if data_cox['Groups'][i]== 'F':
            #data_cox['Groups'][i]= 6
         # One may add these in dependednt on the optimal ammount of clusters