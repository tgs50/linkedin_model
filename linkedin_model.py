import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

s = pd.read_csv('social_media_usage.csv')
ss = s[['income', 'educ2', 'web1h', 'par', 'marital', 'age']]
ss = ss[~((ss['income'] > 9) | 
         (ss['educ2'] > 8) |
         (ss['web1h'] > 2) |
         (ss['par'] > 2) |
         (ss['marital'] > 2) |
         (ss['age'] > 98) |
          ss.isna().any(axis=1))]
ss.rename(columns={'web1h': 'sm_li'}, inplace=True)
y = ss['sm_li']
X = ss[['income', 'educ2', 'par', 'marital', 'age']]
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987)
lr = model = LogisticRegression(class_weight='balanced', random_state=987)
lr.fit(X_train, y_train)

st.title("LinkedIn User Prediction")

st.title("LinkedIn User Prediction")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, step=1)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "Doctorate"])
num_companies = st.number_input("Number of Companies Worked", min_value=0, max_value=20, step=1)
satisfaction = st.slider("Job Satisfaction (1-5)", min_value=1, max_value=5)
