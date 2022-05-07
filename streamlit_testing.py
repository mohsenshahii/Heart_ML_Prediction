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

from digitizer import digitizer

#introducing Dataset
heart_df = pd.read_csv('heart_2020_cleaned.csv')
st.write('Heart Disease prediction')
st.write(heart_df)

heart_df_dg = digitizer(heart_df)

heart_corr = heart_df_dg.corr()
fig,ax=plt.subplots(figsize=(10,6))
sb.heatmap(heart_corr,annot=True,ax=ax)
st.write(fig)

 # cd source\repos\streamlit testing\streamlit testing\Heart
 # streamlit run streamlit_testing.py