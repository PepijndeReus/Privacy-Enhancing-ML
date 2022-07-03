# import modules
import pandas as pd
import os
import numpy as np
import pyRAPL

from sklearn.preprocessing import MinMaxScaler

"""
Please note that the data has been cleaned (=no missing/NaN values) in advance.
This is done by Cleaning.py
"""

# change directory
os.chdir('Adult')

# measure energy consumption for adult
def measure_prepros_adult():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../Energy/Measurement_preprocessing_adult.csv')

    with pyRAPL.Measurement('adult', output=csv_output):
        # load data
        adult_train = pd.read_csv('Adult_train.csv')
        adult_val = pd.read_csv('Adult_val.csv')

        def preprocess_dataset(adult):
            # make binary labels for income column
            adult['income'] = adult['income'].str.replace('<=50K', '0')
            adult['income'] = adult['income'].str.replace('>50K', '1')
            adult['income'] = adult['income'].astype(int)

            # make array with labels, remove labels from dataframe
            labels = adult['income'].copy()
            # labels = np.array(labels)
            adult = adult.drop(['income'], axis=1)

            # use Min-max scaling for continuous features
            adult[['age','capital_gain','capital_loss','hr_per_week']] = MinMaxScaler().fit_transform(adult[['age','capital_gain','capital_loss','hr_per_week']])

            # use One-hot encoding for categorial features
            adult = pd.get_dummies(adult,columns = ['type_employer','education','marital','occupation','relationship','race','sex','country'])
            
            return adult, labels

        # apply preprocessing to training and validation set
        adult_train, labels_train = preprocess_dataset(adult_train)
        adult_val, labels_val = preprocess_dataset(adult_val)

        set(adult_train.columns).difference(adult_val.columns)

        # now save the .csv files
        adult_train.to_csv('Adult_train_data.csv', index=False)
        labels_train.to_csv('Adult_train_labels.csv', index=False)
        # print("Adult training set saved to csv!\n")

        adult_val.to_csv('Adult_val_data.csv', index=False)
        labels_val.to_csv('Adult_val_labels.csv', index=False)
        # print("Adult validation set saved to csv!")

    csv_output.save()

for _ in range(10):
	measure_prepros_adult()

print("Measurement of adult preprocessing saved to csv.")

# and now for the student performance set
# set path
os.chdir('../Student')

# start measurement energy consumption
def measure_prepros_student():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../Energy/Measurement_preprocessing_student.csv')

    with pyRAPL.Measurement('student', output=csv_output):
        # load data
        student = pd.read_csv('student-por.csv')

        # for testing if conversion goes right
        # print(f"Number of fails before processing: {len(student[student.G3 < 10])}")

        # convert student grade to pass or fail
        student.loc[student['G3'] < 10, 'G3'] = 0
        student.loc[student['G3'] > 9, 'G3'] = 1
        student['G3'] = student['G3'].astype(int)

        # for testing
        # print(f"Number of fails after processing: {len(student[student.G3 == 0])}\n")

        # use Min-max scaling for continuous features
        student[['age','absences','G1','G2']] = MinMaxScaler().fit_transform(student[['age','absences','G1','G2']])

        # split training data and label
        student_data = student.loc[:,student.columns != 'G3']
        student_target = student['G3']

        # use One-hot encoding for categorial features
        columns = student_data.columns.values.tolist()
        continous_columns = ['age','absences','G1','G2']
        categorial_columns = [feature for feature in columns if feature not in continous_columns]
        student_data = pd.get_dummies(student_data, columns = categorial_columns)

        # split into training and validation set, same ratio as Adult set
        student_train = student_data[:int(len(student_data) * (2/3))]
        student_val = student_data[int(len(student_data) * (2/3)):]
        grade_train = student_target[:int(len(student_data) * (2/3))]
        grade_val = student_target[int(len(student_data) * (2/3)):]

        # check if sets are equal
        # print(f"Length of training set: {len(student_train)}, length of validation set: {len(student_val)}.")
        # print(f"Training set and labels have same length: {len(student_train) == len(grade_train)}")
        # print(f"Validation set and labels have same length: {len(student_val) == len(grade_val)}")

        # now save the .csv files
        student_train.to_csv('student_train_data.csv', index=False)
        grade_train.to_csv('student_train_grade.csv', index=False)
        # print("Student performance training set saved to csv!\n")

        # and for the validation set
        student_val.to_csv('student_val_data.csv', index=False)
        grade_val.to_csv('student_val_grade.csv', index=False)
        # print("Student performance validation set saved to csv!")

    # save energy report
    csv_output.save()

for _ in range(10):
	measure_prepros_student()

print("Measurement of student preprocessing saved to csv.")