"""
This file uses pyRAPL to measure the energyconsumption of the k-nearest neighbours machine learning model on both the adult and student data set.
"""

# import modules
import pandas as pd
import numpy as np
import os
import pyRAPL

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def measure_knn_adult():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../../Benchmark/Performance/Energy_knn.csv')

    with pyRAPL.Measurement('Adult', output=csv_output):
        # load data
        adult_train_data = pd.read_csv('Adult_train_data.csv')
        adult_train_labels = pd.read_csv('Adult_train_labels.csv')
        adult_val_data = pd.read_csv('Adult_val_data.csv')
        adult_val_labels = pd.read_csv('Adult_val_labels.csv')

        # make and fit kNN model
        NeighbourModel = KNeighborsClassifier(n_neighbors=5)
        NeighbourModel.fit(adult_train_data, adult_train_labels.values.ravel())

        # make report and add to dataframe
        report = classification_report(adult_val_labels, NeighbourModel.predict(adult_val_data), output_dict=True)

        # save data to csv file
        df = pd.DataFrame([report['accuracy']])
        df.columns = ['Accuracy']
        df.to_csv('../../Benchmark/Performance/Accuracy_knn_adult.csv', index=False)

    # save energy consumption
    csv_output.save()

def measure_knn_student():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../../Benchmark/Performance/Energy_knn.csv')

    with pyRAPL.Measurement('Student', output=csv_output):
        # load data
        student_train_data = pd.read_csv('student_train_data.csv')
        student_train_grade = pd.read_csv('student_train_grade.csv')
        student_val_data = pd.read_csv('student_val_data.csv')
        student_val_grade = pd.read_csv('student_val_grade.csv')

        # make and fit kNN model for k = 5
        NeighbourModel = KNeighborsClassifier(n_neighbors=5)
        NeighbourModel.fit(student_train_data, student_train_grade.values.ravel())

        # make report and add to dataframe
        report = classification_report(student_val_grade, NeighbourModel.predict(student_val_data), output_dict=True)

        # save data to csv file
        df = pd.DataFrame([report['accuracy']])
        df.columns = ['Accuracy']
        df.to_csv('../../Benchmark/Performance/Accuracy_knn_student.csv', index=False, mode='a')

    # save energy consumption
    csv_output.save()

if __name__ == '__main__':
    # change directory for Adult data set
    os.chdir('../Data/Adult')
    print("Heads up: this will take a while (+/- 2 mins)")
    for _ in range(1,11):
        measure_knn_adult()
        # report progress
        print(f"Adult: Measurement {_} of 10 is saved to csv.")
    
    # change directory for Student data set
    os.chdir('../Student')
    for _ in range(1,11):
        measure_knn_student()
        # report progress
        print(f"Student: Measurement {_} of 10 is saved to csv.")