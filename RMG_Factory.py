# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:45:21 2023

@author: njaeg
"""
import pandas as pd
import statsmodels.api as sm
from linearmodels import IV2SLS

csv_file_path = '/Users/njaeg/OneDrive - University of Guelph/Advanced Econometrics'
df = pd.read_csv("/Users/njaeg/OneDrive - University of Guelph/Advanced Econometrics/ECON4640_guat_encovi_children_indigenous.csv")

df.dropna(inplace=True)
 
#QUESTION 1

#Analyzing population gender means over time
df['boy'] = df['sexo'].apply(lambda x: 1 if x == 'masculino' else 0)
df['girl'] = df['sexo'].apply(lambda x: 1 if x == 'femenino ' else 0)

RMG_boy = df.groupby('year')['boy'].mean()
print(RMG_boy)

RMG_girl = df.groupby('year')['girl'].mean()
print(RMG_girl)

#Question 2

#printing employment menas grouped by sex and year
RMG_EMP = df.groupby(['sexo', 'year'])['RMG'].mean()
print(RMG_EMP)

#QUESTION 3

#defining a function to run an OLS model with specified variables
def run_ols_regression(df, dependent_variable, control_variables=None):
    # Specify the dependent variable
    Y = df[dependent_variable]
    
    # Specify the independent variable matrix
    X = df[['RMG']]  # 'RMG' is the treatment variable; add control variables as needed
    if control_variables is not None:
        X = pd.concat([X, df[control_variables]], axis=1)

    # Fit the OLS model
    model = sm.OLS(Y, sm.add_constant(X)).fit()
    
    if model.mse_model is not None:  # Check if the model converged successfully
        # Print the summary
        print(model.summary())
    else:
        print(f"Model estimation failed for {dependent_variable}.")

#Running first set of OLS models for dependent variables of interest
print("\n\nMinutes_work Model\n")        
run_ols_regression(df, dependent_variable='minutes_work')

print("\n\nMinutes_Study Model\n")
run_ols_regression(df, dependent_variable='minutes_study')

print("\n\nMinutes_bbsit Model\n")
run_ols_regression(df, dependent_variable='minutes_bbsit')

print("\n\nMinutes_hprod Model\n")
run_ols_regression(df, dependent_variable='minutes_hprod')


#Running same ols models controlling for year and factory
print("\n\nMinutes_work controlled Model\n")        
run_ols_regression(df, dependent_variable='minutes_work', control_variables=['year', 'FACTORY'])

print("\n\nMinutes_study controlled Model\n")        
run_ols_regression(df, dependent_variable='minutes_study', control_variables=['year', 'FACTORY'])

print("\n\nMinutes_bbsit controlled Model\n")        
run_ols_regression(df, dependent_variable='minutes_bbsit', control_variables=['year', 'FACTORY'])

print("\n\nMinutes_hprod controlled Model\n")        
run_ols_regression(df, dependent_variable='minutes_hprod', control_variables=['year', 'FACTORY'])


#QUESTION 5

#CREATING AN INTERACTION TERM 'fac_age' 
#Term is a multiplicative interaction term between participant age and dummy variable indicative of RMG factory in participants region

fac_age = df['FACTORY']*df['age']
df['fac_age'] = fac_age

X = df[['FACTORY', 'age', 'fac_age']]
y = df['RMG']

X = sm.add_constant(X)  # Adding a constant term for the intercept
model = sm.OLS(y, X).fit()
print(model.summary())


#QUESTION 6

# Defining function to run IV2SLS model
def run_iv2sls_regression(df, dependent_variable, iv_variable, exog_variables=None):
    # Specify the dependent variable
    Y = df[dependent_variable]
    
    # Specify the endogenous and instrumental variable matrices
    endog = df[['RMG']]  # Endogenous variable
    instruments = df[iv_variable]  # Instrumental variable
    
    # Specify the exogenous variable matrix

    if exog_variables is not None:
        exog = sm.add_constant(df[exog_variables])
    else:
        # Adding constant term if no exogenous variables specified
        exog = sm.add_constant(df[[iv_variable]])
        exog = exog.drop(columns=[iv_variable])
        
    # Fit the IV2SLS model
    model = IV2SLS(dependent=Y, exog=exog, endog=endog, instruments=instruments).fit()
    
    # Return the model
    return model

#No control
dependent_variables = ['minutes_work', 'minutes_study', 'minutes_bbsit', 'minutes_hprod']
for dependent_var in dependent_variables:
    print(f"\nIV2SLS Regression with '{dependent_var}' as the dependent variable\n")
    model = run_iv2sls_regression(df, dependent_variable=dependent_var, iv_variable='fac_age')
    print(model.summary)

# Controlled for year
dependent_variables = ['minutes_work', 'minutes_study', 'minutes_bbsit', 'minutes_hprod']
for dependent_var in dependent_variables:
    print(f"\nIV2SLS Regression with '{dependent_var}' as the dependent variable\n")
    model = run_iv2sls_regression(df, dependent_variable=dependent_var, iv_variable='fac_age', exog_variables='year')
    print(model.summary)
    
    
    
    
    



