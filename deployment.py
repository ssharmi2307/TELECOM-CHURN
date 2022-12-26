# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:03:06 2022

@author: Gopinath
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title('TELECOM CHURN ANALYSIS')
st.sidebar.header('User Input Parameters')

def user_input_features():
    AREA_CODE = st.sidebar.selectbox("area code",('408','415','510'))
    ACCOUNT_LENGTH = st.sidebar.number_input("Enter the ACCOUNT LENGTH")
    VOICE_PLAN = st.sidebar.selectbox("voice plan",('1','0'))
    VOICE_MESSAGES = st.sidebar.number_input("Enter the number of voice messages")
    INTERNATIONAL_PLAN = st.sidebar.selectbox('international plan',('1','0'))
    INTERNATIONAL_MINUTES = st.sidebar.number_input("Enter the total international minutes")
    INTERNATIONAL_CALLS=st.slider('Total international calls',min_value=0,max_value=25,step=1)
    DAY_MINUTES = st.sidebar.number_input("Enter total day minutes")
    DAY_CALLS=st.slider('Total day calls',min_value=0,max_value=165,step=1)
    EVENING_MINUTES = st.sidebar.number_input("Enter total evening minutes")
    EVENING_CALLS=st.slider('Total evening calls',min_value=0,max_value=170,step=1)
    NIGHT_MINUTES = st.sidebar.number_input("Enter total night minutes")
    NIGHT_CALLS=st.slider('Total night calls',min_value=0,max_value=175,step=1)
    CUSTOMER_CALLS = st.slider("Number of customer calls",min_value=0,max_value=10,step=1)
    data = {'area code':AREA_CODE,
            'account length':ACCOUNT_LENGTH,
            'voice mail plan':VOICE_PLAN,
            'number vmail messages':VOICE_MESSAGES,
            'international plan':INTERNATIONAL_PLAN,
            'total intl minutes':INTERNATIONAL_MINUTES,
            'total intl calls':INTERNATIONAL_CALLS,
            'total day minutes':DAY_MINUTES,
            'total day calls':DAY_CALLS,
            'total eve minutes':EVENING_MINUTES,
            'total eve calls':EVENING_CALLS,
            'total night minutes':NIGHT_MINUTES,
            'total night calls':NIGHT_CALLS,
            'customer service calls':CUSTOMER_CALLS}
    features = pd.DataFrame(data,index = [0])
    return features

df = user_input_features()


churn = pd.read_excel("data1.xlsx")
df["area code"] = df["area code"].astype('int')
df["voice mail plan"] = df["voice mail plan"] .astype('int')
df["international plan"] = df["international plan"] .astype('int')


from sklearn.model_selection import train_test_split
X = churn.iloc[:,:14]
y = churn.iloc[:,14]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from imblearn.over_sampling import SMOTE
Smo = SMOTE(random_state=101)
X_train_res, y_train_res = Smo.fit_resample(X_train,y_train)

#.XGboost Classifier
from xgboost import XGBClassifier
#model development
xgb_model=XGBClassifier().fit(X_train_res, y_train_res)

prediction = xgb_model.predict(df)
prediction_proba = xgb_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes, The Customer will CHURN' if prediction_proba[0][1] > 0.5 else 'No, The Customer will NOT CHURN')
st.subheader('Prediction Probability')
st.write(prediction_proba)
