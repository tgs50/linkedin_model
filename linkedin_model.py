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
age = st.number_input("Age", min_value=1, max_value=97, step=1)
education = st.number_input("Age", min_value=1, max_value=8, step=1)
parent = st.number_input("Age", min_value=1, max_value=2, step=1)
marital = st.number_input("Age", min_value=1, max_value=6, step=1)
income = st.number_input("Age", min_value=1, max_value=9, step=1)

if st.button("Predict LinkedIn User"):
    # Create input array for prediction
    user_data = np.array([[income, education, parent, marital, age]])
    
    # Make prediction and compute probability
    prediction = lr.predict(user_data)
    probability = lr.predict_proba(user_data)[:, 1]  # Probability for class 1 (LinkedIn user)
    
    # Display the results
    if prediction == 1:
        st.success(f"The person is predicted to be a LinkedIn user with a probability of {probability[0]:.2f}.")
    else:
        st.info(f"The person is predicted to NOT be a LinkedIn user with a probability of {probability[0]:.2f}.")

