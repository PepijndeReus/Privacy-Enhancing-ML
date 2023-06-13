# import modules
import pandas as pd
import pyRAPL

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

"""
Please note:
This file runs on the anonymised data generated by k-anonymity.py
"""

# measure energy consumption for adult
def measure_prepros_adult(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Measurement_adult_' + str(k) + '_preprocessing.csv')

    with pyRAPL.Measurement('adult', output=csv_output):
        # load data
        df = pd.read_csv('output/adult_' + str(k) + '.csv', delimiter=';')

        # filter suppressed columns (*)
        df = df.loc[:, df.nunique() != 1]
        
        # filter suppressed rows (*)
        df = df.loc[df['type_employer'] != "*"]
        
        # make binary labels for income column
        df['income'] = df['income'].str.replace('<=50K', '0')
        df['income'] = df['income'].str.replace('>50K', '1')
        df['income'] = df['income'].astype(int)
        
        # use get_dummies for categorical columns
        # k = 3 includes occupation, k=10 and k=27 do not
        if k == 3:
            df2 = pd.get_dummies(df,columns = ['capital_gain','capital_loss','type_employer','education','marital','occupation','relationship','race','sex', 'country'])
        else:
            df2 = pd.get_dummies(df,columns = ['capital_gain','capital_loss','type_employer','education','marital','relationship','race','sex', 'country'])
        
        df_data = df2.loc[:,df2.columns != 'income']
        df_target = df['income']

        x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.333)

        # now save the .csv files
        x_train.to_csv('output/processed/Adult_' + str(k) + '_train_data.csv', index=False)
        y_train.to_csv('output/processed/Adult_' + str(k) + '_train_labels.csv', index=False)
        x_test.to_csv('output/processed/Adult_' + str(k) + '_val_data.csv', index=False)
        y_test.to_csv('output/processed/Adult_' + str(k) + '_val_labels.csv', index=False)

    csv_output.save()

# and now for the student performance set
# start measurement energy consumption
def measure_prepros_student(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Measurement_student_' + str(k) + 'preprocessing.csv')

    with pyRAPL.Measurement('student', output=csv_output):
        # load data
        student = pd.read_csv("output/student_" + str(k) + ".csv", delimiter=';')

        # filter supressed columns
        student = student.loc[:, student.nunique() != 1]

        # filter supressed rows
        student = student.loc[student['schoolsup'] != "*"]

        # convert student grade to pass or fail
        student.loc[student['G3'] < 10, 'G3'] = 0
        student.loc[student['G3'] > 9, 'G3'] = 1
        student['G3'] = student['G3'].astype(int)

        if k == 3:
            student2 = pd.get_dummies(student,columns = ['address','Pstatus','schoolsup','famsup','paid','nursery','higher','internet','romantic'])
        if k == 10:
            student2 = pd.get_dummies(student,columns = ['Pstatus','schoolsup','paid','nursery','higher','internet','romantic'])
        if k == 27:
            student2 = pd.get_dummies(student,columns = ['school','guardian','schoolsup','paid','internet','romantic'])
        
        # split training data and label
        df_data = student2.loc[:,student2.columns != 'G3']
        df_target = student['G3']

        x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.333)

        # now save the .csv files
        x_train.to_csv('output/processed/Student_' + str(k) + '_train_data.csv', index=False)
        y_train.to_csv('output/processed/Student_' + str(k) + '_train_labels.csv', index=False)
        x_test.to_csv('output/processed/Student_' + str(k) + '_val_data.csv', index=False)
        y_test.to_csv('output/processed/Student_' + str(k) + '_val_labels.csv', index=False)

    # save energy report
    csv_output.save()

if __name__ == '__main__':
    for _ in range(1,11):
        k_vals = [3, 10, 27]
        for k in k_vals:
            measure_prepros_adult(k)
            # report progress
            print(f"Adult {k}: Measurement {_} of 10 is saved to csv.")
        
        for k in k_vals:
            measure_prepros_student(k)
            # report progress
            print(f"Student {k}: Measurement {_} of 10 is saved to csv.")