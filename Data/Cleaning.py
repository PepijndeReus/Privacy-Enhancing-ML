"""
Data cleaning for adult
"""

# import modules
import pandas as pd
import os
import numpy as np
import pyRAPL

# change directory
os.chdir('Adult')

# measure energy consumption
pyRAPL.setup()
meter = pyRAPL.Measurement('Cleaning_adult')
csv_output = pyRAPL.outputs.CSVOutput('cleaning_adult_energy.csv')
meter.begin()

# load data, use columns from adult.names file
columns = ["age", "type_employer", "fnlwgt", "education", "education_num","marital", "occupation", "relationship", "race","sex","capital_gain", "capital_loss", "hr_per_week","country", "income"]
adult = pd.read_csv("adult.data", sep=",\s",names=columns,na_values=["?"], engine='python')
print(f"Length before cleaning: {len(adult)}")

# delete unnecessary columns
adult = adult.drop('education_num', axis=1)
adult = adult.drop('fnlwgt', axis=1)

# drop NA values from data set
adult = adult.dropna()
print(f"Length after cleaning: {len(adult)}")

adult.to_csv('Adult_train.csv', index=False)
print("Training set saved!\n")

#################### and now for the validation set ####################

# load data, use columns from adult.names file
columns = ["age", "type_employer", "fnlwgt", "education", "education_num","marital", "occupation", "relationship", "race","sex","capital_gain", "capital_loss", "hr_per_week","country", "income"]
adult = pd.read_csv("adult.test", sep=",\s",names=columns,na_values=["?"], engine='python')
print(f"Length before cleaning: {len(adult)}")

# delete unnecessary columns
adult = adult.drop('education_num', axis=1)
adult = adult.drop('fnlwgt', axis=1)
adult['income'] = adult['income'].str.replace('<=50K.', '<=50K')
adult['income'] = adult['income'].str.replace('>50K.', '>50K')

# drop NA values from data set
adult = adult.dropna()
print(f"Length after cleaning: {len(adult)}")

adult.to_csv('Adult_val.csv', index=False)
print("Validation set saved!")

meter.end()
# csv_output.save()
meter.export(csv_output)
