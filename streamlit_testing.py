###
## import libraries
###
from ctypes import alignment
import streamlit as st
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.express as px
from altair import *
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

from new_lib import *
from description_files import *

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

###
## Putting Dataset into a variable, Deleting duplicated rows and down sampling
###
heart_df = pd.read_csv("heart_2020_cleaned.csv")
heart_df = del_duplicated(heart_df)
down_sampled_heart_df = down_sampler(heart_df)


###
## Turning the original data into numerical with digitizer func. to prepare them for ML models
###
heart_df_digitized = digitizer_manual(heart_df)

add_selectbox = st.sidebar.selectbox(
    "What would you like to see?", 
    ("Dataset", "Prediction models", "Charts")
)


# Using "with" notation
if add_selectbox == "Dataset":
    st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Dataset</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:24px; text-align: justify; color: red; '>What topic does the dataset cover?</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + dataset_topic + "</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:24px; text-align: justify; color: red; '> Where did the dataset come from and what treatments did it undergo?</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + dataset_origin + "</p>", unsafe_allow_html=True)
    st.dataframe(heart_df)
    st.markdown("<br><br><p style='font-size:24px; text-align: left; color: red; '>Before Downsampling:</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + target_desc + "</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:18px; text-align: left; color: brown;'>Proportion of observations having unbalenced percentage of 'Yes' and 'No' target variable</p>", unsafe_allow_html=True)
    heart_disease_proportions = heart_df['HeartDisease'].value_counts()
    labels = 'No Heart Disease', 'Heart Disease' 
    fig = px.pie(heart_disease_proportions, values = 'HeartDisease', names = labels, color_discrete_sequence=px.colors.sequential.RdBu)
    st.write(fig)

    st.markdown("<br><br><p style='font-size:24px; text-align: left; color: red; '>After Downsampling:</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; text-align: left; color: brown; '>Python code of downsampling process</p>", unsafe_allow_html=True)
    st.code(down_sampling_algorithm, language='python')
    st.markdown("<p style='font-size:18px; text-align: left; color: brown;'>After downsampling: Balenced percentage of 'Yes' and 'No' target variable</p>", unsafe_allow_html=True)
    heart_disease_proportions = down_sampled_heart_df["HeartDisease"].value_counts()
    labels = 'No Heart Disease', 'Heart Disease' 
    fig = px.pie(heart_disease_proportions, values = 'HeartDisease', names = labels, color_discrete_sequence=px.colors.sequential.RdBu)
    st.write(fig)
    st.markdown("<p style='font-size:18px; text-align: left; color: black; '>After downsampling the number of rows decreases to <b style='font-size:18px; text-align: left; color: red; '>"+ str(len(down_sampled_heart_df['HeartDisease']))+ " </b>rows that are half 'Yes' and half 'No'</p>", unsafe_allow_html=True)
    

elif add_selectbox == "Prediction models":
    st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease prediction</h1>", unsafe_allow_html=True)
    choice = st.sidebar.radio("Work on Downsampled data?",('YES', 'NO'))
    if choice == 'YES':
        ###
        ## Turning the downsampled categorical data into numerical with digitizer func. to prepare them for ML models
        ###
        heart_df_digitized = digitizer(down_sampled_heart_df)
        st.sidebar.write('The choosen model will apply on Downsampled data!')
    else:
        st.sidebar.write('The choosen model will apply on Original data!')


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
    ## Feature Extraction:
    ## with Linear Discriminant Analysis (LDA). LDA is a supervised learning classifier which means it requires both the features and the labels (or X and y).
    ## with Principal Component Analysis (PCA)
    ## with Isomap. ISOmap is very time consuming so we prefer to not use it!
    ###
    option = st.selectbox(
        "Which feature extraction method would you prefer?",
        ("LDA", "PCA", "ISO", "No Feature extraction"),
    )
    if option == "LDA":
        with st.expander("See explanation"):
             st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + lda_desc + "</p>", unsafe_allow_html=True)
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(x_scaled, y)
        x_lda = lda.transform(x_scaled)
        feature_extracted_x = x_lda
    elif option == "PCA":
        with st.expander("See explanation"):
             st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + pca_desc + "</p>", unsafe_allow_html=True)
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x_scaled)
        feature_extracted_x = x_pca
    elif option == "No Feature extraction":
        feature_extracted_x = x_scaled
    else:
        with st.expander("See explanation"):
             st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + iso_desc + "</p>", unsafe_allow_html=True)
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
        with st.expander("See explanation"):
             st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + adaboost_desc + "</p>", unsafe_allow_html=True)
        model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    elif option2 == "GaussianNB":
        with st.expander("See explanation"):
             st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + gaussian_NB_desc + "</p>", unsafe_allow_html=True)
        model = GaussianNB()
    else:
        with st.expander("See explanation"):
             st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + decision_tree_desc + "</p>", unsafe_allow_html=True)
             st.image(decision_tree_image)
        model = DecisionTreeClassifier()
    accuracy, precision = apply_pred_model(feature_extracted_x, y, model)
    st.markdown("<div style='font-size:18px; text-align: left; color: black; '>The Result of applying " 
        + "<b style='font-size:18px; text-align: left; color: red'>" + option2 + "</b>"
        + " model on feature extracted data using "
        + "<b style='font-size:18px; text-align: left; color: red'>" + option + "</b>"
        + " is "
        + "<br>Accuracy: " + accuracy + "<br>Precision: " + precision +"</div>", unsafe_allow_html=True)

elif add_selectbox == "Charts":
    st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Diagrams</h1>", unsafe_allow_html=True)
    ###
    ## Showing some important diagrams of Dataset
    ###
    ## Histogram of SleepTime of participants
    st.markdown("<p style='text-align: left; color: red;'>Bar Chart of SleepTime Variable:</p>", unsafe_allow_html=True)
    st.bar_chart(heart_df["SleepTime"].value_counts().sort_index())
    st.markdown("<p style='text-align: justify; color: black;'>On average, how many hours of sleep does each person get in a 24-hour period?</p><br>", unsafe_allow_html=True)
    ## Bar chart of Mental Health of participants
    st.markdown("<p style='text-align: left; color: red;'>Bar Chart of MentalHealth Variable:</p>", unsafe_allow_html=True)
    st.write(len(heart_df))
    st.bar_chart(heart_df["MentalHealth"].value_counts().sort_index()/len(heart_df))
    st.markdown("<p style='text-align: justify; color: black;'>Thinking about the participants mental health, for how many days during the past 30 days was their mental health not good?</p><br>", unsafe_allow_html=True)
    ## Bar chart of Mental Health of participants having enough sleep
    st.markdown("<p style='text-align: left; color: red;'>Bar Chart of MentalHealth Variable for participants having enough sleep:</p>", unsafe_allow_html=True)
    more_7_mask = heart_df['SleepTime'] > 7
    less_10_mask = heart_df['SleepTime'] < 10
    st.write(len(heart_df[more_7_mask & less_10_mask]))
    st.bar_chart(heart_df[more_7_mask & less_10_mask]['MentalHealth'].value_counts().sort_index()/len(heart_df[more_7_mask & less_10_mask]))
    st.markdown("<p style='text-align: justify; color: black;'>Participants having 7,8 or 9 hours sleep a day have relatively fewer number of mental problems during last 30 days. As for overall partcipants the percentage of zero mental problems is 62% while the same pecentage for participants having enough sleep it's 68% </p><br>", unsafe_allow_html=True)
    
    ## Drawing Line graph of Sleep Time and Mental health in the same figure
    sleeping_time = pd.DataFrame(heart_df["SleepTime"].value_counts().sort_index())
    mental_health = pd.DataFrame(heart_df["MentalHealth"].value_counts().sort_index())
    result = pd.concat([sleeping_time, mental_health], axis=1)
    st.line_chart(result)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>" + sleep_mental_corr + "<hr></p>", unsafe_allow_html=True)


    ## Bar chart of Physical Health of participants
    st.markdown("<p style='text-align: left; color: red;'>Bar Chart of PhysicalHealth Variable:</p>", unsafe_allow_html=True)
    st.bar_chart(heart_df['PhysicalHealth'].value_counts().sort_index())
    st.markdown("<p style='text-align: left; color: black;'>The suprizing fact is among the people claiming that during the past 30 days had zero physical difficulty the biggest proportions are belong to elderly people!</p>", unsafe_allow_html=True)
    healthy_mask =  heart_df['PhysicalHealth'] == 0
    age_category = heart_df[healthy_mask]['AgeCategory'].value_counts()
    fig = px.pie(age_category, values = 'AgeCategory',names = age_category.index, color_discrete_sequence=px.colors.sequential.RdBu)
    st.write(fig)
    st.markdown("<p style='text-align: justify; color: black;'>Now thinking about the participants physical health, which includes physical illness and injury, for how many days during the past 30?</p><br>", unsafe_allow_html=True)

    ##Physical Health (the number of injuries and illnesses during the past 30 days)
    ##and GenHealth(General Health) have a strong negative correlation we show this
    ##by a scatter plot
    st.markdown("<p style='text-align: left; color: red;'>Scatter plot of Physical Health variable:</p>", unsafe_allow_html=True)
    font = {'family' : 'normal',
        'size'   : 3}

    matplotlib.rc('font', **font)
    temp = heart_df.groupby("PhysicalHealth").count()
    fig = plt.figure(figsize=(2, 1))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        temp.index, temp["GenHealth"],
        s = 0.1
    )
    ax.set_xlabel("Number of days having health problem during last 30 days")
    ax.set_ylabel("General health")
    st.write(fig)
    st.markdown("<p style='text-align: justify; color: black;'> Physical Health (the number of injuries and illnesses during the past 30 days) and GenHealth(General Health) have a strong negative correlation we show this by a scatter plot</p><br><hr>", unsafe_allow_html=True)
    ## correlation between Sleep Time and Mental Health.
    occurance_ = heart_df["SleepTime"].value_counts()
    normalized_ = occurance_[heart_df["SleepTime"]] / len(heart_df)
    

    ## Correlation of Dataset's different attributes
    st.markdown("<p style='text-align: left; color: red;'>Heatmap of correlations between different variables of Dataset:</p>", unsafe_allow_html=True)
    heart_corr = heart_df_digitized.corr()
    fig = px.imshow(heart_corr, aspect="auto", width=600, height=600)
    st.write(fig)
    st.markdown("<p style='text-align: justify; color: black;'>" + correlations + "</p><br><hr>", unsafe_allow_html=True)



    ## Pie charting the proportion of Smokers to Non-smokers
    st.markdown("<p style='text-align: left; color: red;'>Percentage of Smoking and Non Smoking people</p>", unsafe_allow_html=True)
    smoking_proportion = heart_df["Smoking"].value_counts()
    labels = 'Non Smoking', 'Smoking' 
    fig = px.pie(smoking_proportion, values = 'Smoking', names = labels, color_discrete_sequence=px.colors.sequential.RdBu)
    st.write(fig)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>Have he\she smoked at least 100 cigarettes in his\her entire life? [Note: 5 packs = 100 cigarettes]</p>", unsafe_allow_html=True)
    ## Pie charting the proportion of smokers among different age categories
    st.markdown("<p style='text-align: left; color: red;'>Percentage of Smoking people in different Age Categories</p>", unsafe_allow_html=True)
    temp = heart_df_digitized.groupby("AgeCategory").sum()
    Age_Cat = heart_df.groupby("AgeCategory").sum()
    fig = px.pie(temp, values = 'Smoking', names = Age_Cat.index, color_discrete_sequence=px.colors.sequential.RdBu )
    st.write(fig)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>Population smoking people among different age categories varies, the biggest number of smokers are in 60-64 and 65-69 age categories and the lowest is related to 18_24 age category <hr></p>", unsafe_allow_html=True)
     ## Pie charting the proportion of smokers among different age categories
    st.markdown("<p style='text-align: left; color: red;'>Percentage of HeartDisease in different Age Categories</p>", unsafe_allow_html=True)
    temp = heart_df_digitized.groupby("AgeCategory").sum()
    Age_Cat = heart_df.groupby("AgeCategory").sum()
    fig = px.pie(temp, values = 'HeartDisease', names = Age_Cat.index, color_discrete_sequence=px.colors.sequential.RdBu )
    st.write(fig)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>Percentage of people having heart disease among different age categories differs, the biggest number of HeartDisease occurs in more than 80 and 70-74 age categories and the lowest is related to 18_24 age category <hr></p>", unsafe_allow_html=True)
    
    ## Pie charting the proportion of people having kidney disease
    st.markdown("<p style='text-align: left; color: red;'>Percentage of people having kidney disease </p>", unsafe_allow_html=True)
    kidney_proportion = heart_df["KidneyDisease"].value_counts()
    labels = "Having no kidney disease", 'Having kidney disease' 
    fig = px.pie(kidney_proportion, values = 'KidneyDisease', names = labels, color_discrete_sequence = px.colors.sequential.RdBu)
    st.write(fig)
    st.markdown("<p style='font-size:17px; text-align: justify; color: black; '>Not including kidney stones, bladder infection or incontinence, were he\she ever told he\she had kidney disease?<hr></p>", unsafe_allow_html=True)


###
## cd source\repos\streamlit testing\streamlit testing\Heart
## streamlit run streamlit_testing.py
###

