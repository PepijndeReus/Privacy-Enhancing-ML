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

def measure_cleaning():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../Energy/Measurement_cleaning.csv')

    with pyRAPL.Measurement('adult', output=csv_output):
        # load data, use columns from adult.names file
        columns = ["age", "type_employer", "fnlwgt", "education", "education_num","marital", "occupation", "relationship", "race","sex","capital_gain", "capital_loss", "hr_per_week","country", "income"]
        adult = pd.read_csv("adult.data", sep=",\s",names=columns,na_values=["?"], engine='python')
        # print(f"Length before cleaning: {len(adult)}")

        # delete unnecessary columns
        adult = adult.drop('education_num', axis=1)
        adult = adult.drop('fnlwgt', axis=1)

        # since only 1 entry for entire set, remove this column
        adult = adult[adult.country != 'Holand-Netherlands']

        # drop NA values from data set
        adult = adult.dropna()
        # print(f"Length after cleaning: {len(adult)}")

        adult.to_csv('Adult_train.csv', index=False)
        # print("Training set saved!\n")

        #################### and now for the validation set ####################

        # load data, use columns from adult.names file
        columns = ["age", "type_employer", "fnlwgt", "education", "education_num","marital", "occupation", "relationship", "race","sex","capital_gain", "capital_loss", "hr_per_week","country", "income"]
        adult = pd.read_csv("adult.test", sep=",\s",names=columns,na_values=["?"], engine='python')
        # print(f"Length before cleaning: {len(adult)}")

        # delete unnecessary columns
        adult = adult.drop('education_num', axis=1)
        adult = adult.drop('fnlwgt', axis=1)
        adult['income'] = adult['income'].str.replace('<=50K.', '<=50K')
        adult['income'] = adult['income'].str.replace('>50K.', '>50K')

        # drop NA values from data set
        adult = adult.dropna()
        # print(f"Length after cleaning: {len(adult)}")

        adult.to_csv('Adult_val.csv', index=False)
        # print("Validation set saved!")

    csv_output.save()

for _ in range(10):
	measure_cleaning()

print("Measurement of adult cleaning saved to csv.")

# change separator for synthetic data later
df = pd.read_csv('../Student/student-por.csv', sep=';')
df.to_csv('../Student/student-por.csv', index=False)
print("Student saved to csv with , delimiter.")