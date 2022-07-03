import pandas as pd
import numpy as np
import os
import pyRAPL

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def measure_logreg_adult():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../../Benchmark/Performance/Energy_logreg.csv')

    with pyRAPL.Measurement('Adult', output=csv_output):
        # load data
        adult_train_data = pd.read_csv('Adult_train_data.csv')
        adult_train_labels = pd.read_csv('Adult_train_labels.csv')
        adult_val_data = pd.read_csv('Adult_val_data.csv')
        adult_val_labels = pd.read_csv('Adult_val_labels.csv')

        # make Logistics Regression model
        LogReg = LogisticRegression(max_iter=1000)
        LogReg.fit(adult_train_data, adult_train_labels.values.ravel())

        # predict and print report
        predictions = LogReg.predict(adult_val_data)
        report = classification_report(adult_val_labels.values.ravel(), predictions, output_dict=True)
        # print(f"The accuracy for the adult data set is: {report['accuracy']}")

        # save results to csv
        df = pd.DataFrame([report['accuracy']])
        df.columns = ['Accuracy']
        df.to_csv('../../Benchmark/Performance/Accuracy_logreg_adult.csv', index=False)
    
    # save energy consumption
    csv_output.save()

def measure_logreg_student():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../../Benchmark/Performance/Energy_logreg.csv')

    with pyRAPL.Measurement('Student', output=csv_output):
        # load data
        student_train_data = pd.read_csv('student_train_data.csv')
        student_train_grade = pd.read_csv('student_train_grade.csv')
        student_val_data = pd.read_csv('student_val_data.csv')
        student_val_grade = pd.read_csv('student_val_grade.csv')

        # make Logistics Regression model
        LogReg = LogisticRegression(max_iter=1000)
        LogReg.fit(student_train_data, student_train_grade.values.ravel())

        # predict and print report
        predictions = LogReg.predict(student_val_data)
        report = classification_report(student_val_grade.values.ravel(), predictions, output_dict=True)
        # print(f"The accuracy for the student data set is: {report['accuracy']}")

        # save results to csv
        df = pd.DataFrame([report['accuracy']])
        df.columns = ['Accuracy']
        df.to_csv('../../Benchmark/Performance/Accuracy_logreg_student.csv', index=False)

    # save energy consumption
    csv_output.save()

if __name__ == '__main__':
    # change directory for Adult data set
    os.chdir('../Data/Adult')
    for _ in range(1,11):
        measure_logreg_adult()
        # report progress
        print(f"Adult: Measurement {_} of 10 is saved to csv.")
    
    # change directory for Student data set
    os.chdir('../Student')
    for _ in range(1,11):
        measure_logreg_student()
        # report progress
        print(f"Student: Measurement {_} of 10 is saved to csv.")