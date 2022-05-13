###
## import libraries
###
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

from digitizer import digitizer
from apply_model import apply_model


###
## Putting Dataset into a variable
###
heart_df = pd.read_csv('heart_2020_cleaned.csv')
st.write('Heart Disease prediction')

###
## Turning the categorical data into numerical with digitizer func. to prepare them for ML models
###
heart_df_digitized = digitizer(heart_df)

###
## Splitting dataset into independent and dependent variables 
###
y = heart_df_digitized['HeartDisease']
x = heart_df_digitized.drop('HeartDisease', axis=1)

###
## Standardizing Data
###
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

###
## Feature Extraction with Principal Component Analysis (PCA)
###
# pca = PCA(n_components=2)
# x_pca = pca.fit_transform(x_scaled)

###
## Feature Extraction with Linear Discriminant Analysis (LDA). LDA is a supervised learning classifier which means it requires both the features and the labels (or X and y). 
###
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x_scaled, y)
x_lda = lda.transform(x_scaled)

###
## Feature Extraction with Isomap. ISOmap is very time consuming so we prefer to not use it!
###
# iso = Isomap(n_components=2)
# x_iso = iso.fit_transform(x_scaled)


###
## Applying the ML model on the data: DecisionTreeClassifier() GaussianNB()
###
model = AdaBoostClassifier(n_estimators=50,learning_rate=1)
st.write(apply_model( x_lda, y, model))


heart_corr = heart_df_digitized.corr()
fig,ax=plt.subplots(figsize=(10,6))
sb.heatmap(heart_corr,annot=True,ax=ax)
st.write(fig)

### 
## cd source\repos\streamlit testing\streamlit testing\Heart
## streamlit run streamlit_testing.py
###