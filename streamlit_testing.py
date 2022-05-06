#introducing needed libraries
import streamlit as st
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier

#introducing Dataset
heart_df = pd.read_csv('heart_2020_cleaned.csv')
st.write('Heart Disease prediction')
st.write(heart_df)


heart_df['Diabetic'].replace('No, borderline diabetes', 1, inplace = True)
heart_df['Diabetic'].replace('Yes', 2, inplace = True)
heart_df['Diabetic'].replace('Yes (during pregnancy)', 3, inplace = True)
heart_df.replace('Yes', 1, inplace = True)
heart_df.replace('No', 0, inplace = True)
heart_df['Sex'].replace(['Female', 'Male'], [0, 1], inplace = True)
heart_df['Race'].replace(['White', 'Hispanic', 'Black', 'Other', 'Asian', 'American Indian/Alaskan Native'], [1, 2, 3, 4, 5, 6] , inplace = True)
heart_df['GenHealth'].replace(['Very good','Good', 'Excellent', 'Fair', 'Poor'], [5, 4, 3, 2, 1], inplace = True)
heart_df['AgeCategory'].replace(['18-24','25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], inplace = True)
 
heart_corr = heart_df.corr()
fig,ax=plt.subplots(figsize=(10,6))
sb.heatmap(heart_corr,annot=True,ax=ax)
st.write(fig)