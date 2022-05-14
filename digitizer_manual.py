import pandas as pd

def digitizer_manual(dataframe):
  '''
     We can also do transformation by "replace" function
  '''
  dataframe_ = dataframe.copy()
  dataframe_['Diabetic'].replace('No, borderline diabetes', 1, inplace = True)
  dataframe_['Diabetic'].replace('Yes', 2, inplace = True)
  dataframe_['Diabetic'].replace('Yes (during pregnancy)', 3, inplace = True)
  dataframe_.replace('Yes', 1, inplace = True)
  dataframe_.replace('No', 0, inplace = True)
  dataframe_['Sex'].replace(['Female', 'Male'], [0, 1], inplace = True)
  dataframe_['Race'].replace(['White', 'Hispanic', 'Black', 'Other', 'Asian', 'American Indian/Alaskan Native'], [1, 2, 3, 4, 5, 6], inplace = True)
  dataframe_['GenHealth'].replace(['Very good','Good', 'Excellent', 'Fair', 'Poor'], [5, 4, 3, 2, 1], inplace = True)
  dataframe_['AgeCategory'].replace(['18-24','25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], inplace = True)
  return dataframe_