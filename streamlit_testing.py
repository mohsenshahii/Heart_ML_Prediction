###
## import libraries
###
import streamlit as st
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from termcolor import colored
from time import time
from scipy.stats.stats import pearsonr
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

from new_lib import del_duplicated
from new_lib import digitizer_manual
from new_lib import digitizer
from new_lib import apply_pred_model


###
## Importing description text files
###
with open('dataset_topic.txt') as f:
    dataset_topic = f.read()
    f.close()
with open('dataset_origin.txt') as f:
    dataset_origin = f.read()
    f.close()
with open('pca_description.txt') as f:
    pca_desc = f.read()
    f.close()
with open('lda_desc.txt') as f:
    lda_desc = f.read()
    f.close()
with open('iso_desc.txt') as f:
    iso_desc = f.read()
    f.close()
###
## Putting Dataset into a variable
###
heart_df = pd.read_csv("heart_2020_cleaned.csv")
st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease prediction</h1>", unsafe_allow_html=True)

add_selectbox = st.sidebar.selectbox(
    "What would you like to see?", ("Dataset", "Prediction models", "Charts")
)

###
## Deleting duplicated rows
###
heart_df = del_duplicated(heart_df)

###
## Turning the categorical data into numerical with digitizer func. to prepare them for ML models
###
heart_df_digitized = digitizer_manual(heart_df)

# Using "with" notation
if add_selectbox == "Dataset":
    st.markdown("<p style='font-size:24px; text-align: left; color: red; '>What topic does the dataset cover?</p>", unsafe_allow_html=True)
    st.write(dataset_topic)
    st.markdown("<p style='font-size:24px; text-align: left; color: red; '> Where did the dataset come from and what treatments did it undergo?</p>", unsafe_allow_html=True)
    st.write(dataset_origin)
    heart_df
elif add_selectbox == "Prediction models":
    ###
    ## Splitting dataset into independent and dependent variables
    ###
    y = heart_df_digitized["HeartDisease"]
    x = heart_df_digitized.drop("HeartDisease", axis=1)
    ###
    ## Standardizing Data
    ###
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    ###
    ## Feature Extraction:
    ## with Linear Discriminant Analysis (LDA). LDA is a supervised learning classifier which means it requires both the features and the labels (or X and y).
    ## with Principal Component Analysis (PCA)
    ## with Isomap. ISOmap is very time consuming so we prefer to not use it!
    ###
    option = st.selectbox(
        "Which feature extraction method would you prefer?",
        ("lda", "pca", "iso", "No one"),
    )
    if option == "lda":
        st.write(lda_desc)
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(x_scaled, y)
        x_lda = lda.transform(x_scaled)
        feature_extracted_x = x_lda
    elif option == "pca":
        st.write(pca_desc)
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x_scaled)
        feature_extracted_x = x_pca
    elif option == "No one":
        feature_extracted_x = x_scaled
    else:
        st.write(iso_desc)
        iso = Isomap(n_components=2)
        x_iso = iso.fit_transform(x_scaled)
        feature_extracted_x = x_iso

    ###
    ## Applying the ML model on the data: DecisionTreeClassifier() GaussianNB()
    ###
    option2 = st.selectbox(
        "Which Machine Learning method would you prefer?",
        ("AdaBoosting", "GaussianNB", "DecisionTree"),
    )
    if option2 == "AdaBoosting":
        model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    elif option2 == "GaussianNB":
        model = GaussianNB()
    else:
        model = DecisionTreeClassifier()
    
    st.markdown("<div style='font-size:18px; text-align: left; color: black; '>The accuracy of applying " 
        + "<b style='font-size:22px; text-align: left; color: red'>" + option2 + "</b>"
        + " on feature extracted data using "
        + "<b style='font-size:22px; text-align: left; color: red'>" + option + "</b>"
        + " is "
        + "<b>" + str(apply_pred_model(feature_extracted_x, y, model)) +"</b></div>", unsafe_allow_html=True)
    #st.write(
    #    "The accuracy of applying " 
    #    + '\033[1m'
    #    + option2
    #    + '\033[1m'
    #    + " on feature extracted data using "
    #    + option
    #    + " is "
    #    + str(apply_pred_model(feature_extracted_x, y, model))
    #)

elif add_selectbox == "Charts":

    ###
    ## Showing some important diagrams of Dataset
    ###
    ## Histogram of SleepTime of participants
    fig, ax = plt.subplots(figsize=(6, 5))
    heart_df_digitized["SleepTime"].hist(bins=40, ax=ax)
    st.write(fig)

    ## Correlation of Dataset's different attributes
    heart_corr = heart_df_digitized.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sb.heatmap(heart_corr, ax=ax)
    st.write(fig)

    ## Physical Health (the number of injuries and illnesses during the past 30 days)
    ## and GenHealth(General Health) have a strong negative correlation we show this
    ## by a scatter plot
    temp = heart_df.groupby("PhysicalHealth").count()
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        temp.index, temp["GenHealth"],
    )
    ax.set_xlabel("Number of days having health problem during last 30 days")
    ax.set_ylabel("General health")
    st.write(fig)

    ## correlation between Sleep Time and Mental Health.
    occurance_ = heart_df["SleepTime"].value_counts()
    normalized_ = occurance_[heart_df["SleepTime"]] / len(heart_df)
    description = """Considering Sleep Time and Mental Health:
                  There should be a positive correlation between Mental 
                  Health and the quantity of occuranse of sleep hours in 
                  the dataset because SleepTime is normaly distributed 
                  and the more normal a value is the more times it is
                  apeared so instead of calculating the correlation between
                  Sleep Time and Mental Health we calculate correlation 
                  between occuranse percentage of a Sleep Time and Mental Health."""
    description
    st.write(pearsonr(heart_df["MentalHealth"], normalized_))

    ## Line graphing the variables of Sleep Time and Mental health in the same figure
    sleeping_time = heart_df["SleepTime"].value_counts().sort_index()
    mental_health = heart_df["MentalHealth"].value_counts().sort_index()

    ## Pie charting the proportion of Smokers to Non-smokers
    smoking_proportions = heart_df["Smoking"].value_counts()
    labels = "Non Smoking", "Smoking"
    fig, ax = plt.subplots()
    ax.pie(smoking_proportions, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Pie")
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    ## Pie charting the proportion of smokers among different age categories
    temp = heart_df_digitized.groupby("AgeCategory").sum()
    Age_Cat = heart_df.groupby("AgeCategory").sum()
    fig, ax = plt.subplots()
    ax.pie(
        temp["Smoking"],
        labels=Age_Cat.index,
        autopct=lambda x: str(x)[:4] + "%",
        textprops={"fontsize": 6},
    )
    ax.set_title("Percentage of Smoking people in different Age Categories")
    st.pyplot(fig)


###
## cd source\repos\streamlit testing\streamlit testing\Heart
## streamlit run streamlit_testing.py
###
