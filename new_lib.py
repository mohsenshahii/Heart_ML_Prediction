from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def apply_pred_model(x, y,model):
  '''
    This function receives the X and y dataset set, a model name, splits into train and test, 
    applies the model on the train data, make prediction after training, 
    and returns the accuracy and precision
  '''
  x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.3, random_state=1)
  model_ = model
  model_.fit(x_train, y_train)
  y_pred = model_.predict(x_test)

  accuracy = accuracy_score(y_test, y_pred)
 
  precision = precision_score(y_test, y_pred)

  return (str(accuracy)[:6],
          str(precision)[:6])
          


def del_duplicated(dataframe):
    df = dataframe.copy()
    df.drop_duplicates(inplace = True)
    new_len = len(df)
    index = pd.Index(range(0,new_len))
    df.set_index(index, inplace = True)
    return df


def digitizer(dataframe):
  '''
     Easier way to transforming categorical data to number was get_dummies function
     This method of digitizing  slightly(1-2 percents) increase the accuracy of ML models
  '''
  dataframe['HeartDisease'].replace(['Yes','No'],[1 , 0], inplace = True)
  digitized = pd.get_dummies(dataframe)
  return digitized


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

def down_sampler(dataframe):
    mask = dataframe['HeartDisease']=='Yes'
    temp = dataframe[mask]
    dataframe_ = dataframe.drop(dataframe[dataframe['HeartDisease']=='Yes'].index)
    temp.index = range(0,len(temp))
    dataframe_.index=range(0,len(dataframe_))
    #Generate 27373 random numbers between 0 and 292422
    randomlist = random.sample(range(0,len(dataframe_) ),len(temp))
    down_sampled = temp.append(dataframe_.loc[randomlist], ignore_index=True)
    down_sampled = down_sampled.sample(frac=1).reset_index(drop=True)
    return down_sampled