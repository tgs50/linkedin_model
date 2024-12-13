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

st.markdown("data defintions:")

st.markdown("age (numeric age): 1 to 97")

st.markdown("1	Less than high school (Grades 1-8 or no formal schooling)
	2	High school incomplete (Grades 9-11 or Grade 12 with NO diploma)
	3	High school graduate (Grade 12 with diploma or GED certificate)
	4	Some college, no degree (includes some community college)
	5	Two-year associate degree from a college or university
	6	Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)
	7	Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)
	8	Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)")

st.markdown("Parent: 1	Yes
	2	No")

st.markdown(current marital status

	1	Married
	2	Living with a partner
	3	Divorced
	4	Separated
	5	Widowed
	6	Never been married")

# User input fields
age = st.number_input("Age", min_value=1, max_value=97, step=1)
education = st.number_input("Education", min_value=1, max_value=8, step=1)
parent = st.number_input("Parent", min_value=1, max_value=2, step=1)
marital = st.number_input("Marital", min_value=1, max_value=6, step=1)
income = st.number_input("Income", min_value=1, max_value=9, step=1)

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

