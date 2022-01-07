# 1. Main libraries
import numpy as np
import pandas as pd

import random
import plotly.express as px

# 2. Splitting the dataset
from sklearn.model_selection import train_test_split

# 3. Models
import catboost as catb

# 4. Quality metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

import pickle

import streamlit as st

header = st.container()

main_body = st.container()

foot = st.container()

with header:
    st.title('Wholesale Region Prediction')

with main_body:
    # markdown scripts
    st.write("""
    ##### This Wholesale Region Prediction model was built using CatBoostClassifier 

    The app will predict sone of the following 3 regions of Russia:

    - **Yekaterinburg**
    - **Moscow**
    - **Other**
    """)
    channel = st.text_input('CHANNEL')
    
    fresh = st.number_input('Annual spending on FRESH products')
    
    milk = st.number_input('Annual spending on MILK products')
    
    grocery = st.number_input('Annual spending on GROCERY products')
    
    frozen = st.number_input('Annual spending on FROZEN products')
    
    det_paper = st.number_input('Annual spending on DETERGENTS and PAPER products')
    
    delic = st.number_input('Annual spending on DELICATESSEN products')

    btn = st.button('Predict Class')
    if btn:
        st.write(
            '........Based on your input, the Region is..........')

        # load the model
        loaded_ClassifierModel = pickle.load(open('wholesale_model.pkl', 'rb'))

        # requires a 2-D array
        mydata = np.array([[channel, fresh, milk, grocery, frozen, det_paper, delic]])
        # st.write(mydata)

        # predict and print it out
        PredictedRegion = loaded_ClassifierModel.predict(mydata)
        st.write(PredictedRegion)
    # else:
    #    st.write('Goodbye')

with foot:
    st.caption('Created by Daria Chermnykh')

# streamlit run main.py
