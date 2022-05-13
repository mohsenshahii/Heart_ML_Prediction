import pandas as pd

def digitizer(dataframe):
  '''
     The easier way to transforming categorical data to number was get_dummies function
     This method of digitizing  slightly(1-2 percents) increase the accuracy of ML models
  '''
  dataframe.replace(['Yes','No'],[1 , 0], inplace = True)
  digitized = pd.get_dummies(dataframe)
  return digitized