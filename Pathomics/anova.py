#!/bin/python3
# Standard library modules.
import math

# Third party modules.
import numpy
import pandas
import IPython
import statsmodels
import statsmodels.formula
import statsmodels.api

# Local modules.

##############################
# FUNCTIONS                  #
##############################
def anova_desc_table(dataSpecGraphGroups, Features, title, dep='Groups',groupList=[0,1,2]):
    IPython.display.display(IPython.display.Latex(r"$\color{blue}{\Large ANOVA\ Table\ feature\ per\ Group}$"))
    anova_mi = pandas.MultiIndex.from_product([[f"Between {dep}", f"Within {dep}", "Total"], ['Sum of Squares', 'df', 'Mean Sqaure', 'F', 'Sig.']])
    anova_df = pandas.DataFrame(columns=anova_mi, index=Features[:-1])
    index = pandas.MultiIndex.from_product([Features[:-1],[0]], names=['Feature','Sig.'])
    columns = pandas.MultiIndex.from_product([[dep],groupList, ['N','Mean','Standard Deviation','Standard Deviation Error','95% Upper Bound Mean','95% Lower Bound Mean']])
    anova_desc_df = pandas.DataFrame(columns=columns, index=index)
    anova_desc_df = anova_desc_df.reset_index(level='Sig.')
    for par in Features[:-1]:
        model_name = statsmodels.formula.api.ols(par+' ~ C('+dep+')', data=dataSpecGraphGroups).fit()
        ano = statsmodels.api.stats.anova_lm(model_name,typ=1)
        ano = pandas.concat([ano, pandas.DataFrame({"df":[ano.df.sum()], "sum_sq":[ano.sum_sq.sum()]}, index=["Total"])])
        ano = ano[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']]
        ano.rename(columns = {"sum_sq": "Sum of Squares", 'mean_sq':'Mean Sqaure', 'PR(>F)':'Sig.'},inplace=True)
        ano.rename(index = {'C('+dep+')': "Between "+dep, 'Residual':'Within '+dep},inplace=True)
        anova_df.at[str(par)] = ano.values.ravel()
        anova_desc_df.loc[par,'Sig.'] = anova_df.loc[str(par),'Between '+dep]['Sig.']
        gb_Groups = dataSpecGraphGroups.groupby([dep])[str(par)].agg(['count', 'mean', 'std', ('sem', sem), ('ci95_hi',lambda x: (numpy.mean(x) + 1.96*numpy.std(x)/math.sqrt(numpy.size(x)))), ('ci95_lo',lambda x: (numpy.mean(x) - 1.96*numpy.std(x)/math.sqrt(numpy.size(x))))])
        anova_desc_df.loc[str(par),'Groups'] = gb_Groups.values.ravel()
    IPython.display.display(anova_desc_df)
    anova_desc_df.to_csv(title+' ANOVA + Descriptive Table - ' +dep +'.csv')
    return anova_desc_df