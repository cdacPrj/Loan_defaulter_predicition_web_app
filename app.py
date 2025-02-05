# importing dependencies
from flask import Flask,request,jsonify,render_template
# import joblib
import numpy as np
# import pandas as pd
import shap
import pickle
import bz2

# load model with joblib 
# model =joblib.load("random_forest_modelv2.pkl")


# Load the compressed model
with bz2.BZ2File("modelbz2.pkl.bz2", "rb") as f:
    model = pickle.load(f)


# input preparation
import numpy as np

def process_form_input(values):
    """
    Converts input list into a numpy array of length 28.
    
    Parameters:
    values (list): List of input values from the form, ordered as:
    ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
     'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasMortgage',
     'HasDependents', 'HasCoSigner', 'Default', 'EmploymentType',
     'MaritalStatus', 'LoanPurpose', 'Education']
    
    Returns:
    np.ndarray: Processed numpy array of shape (28,)
    """
    
    # Initialize an array of zeros with length 28
    features = np.zeros(28)
    
    # Assign numerical values directly
    features[:9] = values[:9]  # ['Age', 'Income', 'LoanAmount', ..., 'DTIRatio']
    
    # Binary categorical features
    features[9] = 1 if values[9] == 'Yes' else 0  # HasMortgage
    features[10] = 1 if values[10] == 'Yes' else 0  # HasDependents
    features[11] = 1 if values[11] == 'Yes' else 0  # HasCoSigner
    
    # Employment Type One-Hot Encoding
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    features[12:16] = [1 if values[12] == emp else 0 for emp in employment_types]
    
    # Marital Status One-Hot Encoding
    marital_statuses = ['Divorced', 'Married', 'Single']
    features[16:19] = [1 if values[13] == status else 0 for status in marital_statuses]
    
    # Loan Purpose One-Hot Encoding
    loan_purposes = ['Auto', 'Business', 'Education', 'Home', 'Other']
    features[19:24] = [1 if values[14] == purpose else 0 for purpose in loan_purposes]
    
    # Education One-Hot Encoding
    education_levels = ["Bachelor's", 'High School', "Master's", 'PhD']
    features[24:28] = [1 if values[15] == level else 0 for level in education_levels]
    
    return features


# fetching feature importance using 
def get_features_imp_using_shap(model,inp):
    explainer = shap.Explainer(model)
    shap_values = explainer(inp)
    return np.abs(shap_values.values).mean(axis=1)
    
    



app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    form_features=[x for x in request.form.values()]
    model_input_features=process_form_input(form_features)

    columnNames=['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
       'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasMortgage',
       'HasDependents', 'HasCoSigner', 'EmploymentType_Full-time',
       'EmploymentType_Part-time', 'EmploymentType_Self-employed',
       'EmploymentType_Unemployed', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'LoanPurpose_Auto',
       'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home',
       'LoanPurpose_Other', "Education_Bachelor's", 'Education_High School',
       "Education_Master's", 'Education_PhD']
    
    prediction = model.predict(model_input_features.reshape(1, -1))
    output = 'Defaulter' if prediction[0] == 1 else 'Not a defaulter'
    
    feature_importance=get_features_imp_using_shap(model,model_input_features)

    # features after one hot encoding
    
    
    feature_importance_dict = dict(zip(columnNames, feature_importance))

    return render_template('result.html', prediction=output, feature_importance=feature_importance_dict)

if __name__ == "__main__":
    app.run(debug=True)