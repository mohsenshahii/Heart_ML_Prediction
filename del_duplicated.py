import pandas as pd

def del_duplicated(dataframe):
    df = dataframe.copy()
    df.drop_duplicates(inplace = True)
    new_len = len(df)
    index = pd.Index(range(0,new_len))
    df.set_index(index, inplace = True)
    return df