import pandas as pd
def digitizer(dataframe):
    dataframe.replace(['Yes','No'],[1 , 0], inplace = True)
    digitized = pd.get_dummies(dataframe)
    y = dataframe['HeartDisease']
    x = dataframe.drop('HeartDisease', axis=1)
    return digitized