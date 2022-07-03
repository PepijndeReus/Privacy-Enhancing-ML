import os
import pyRAPL
import pandas as pd

# import modules
from sklearn.neighbors import KNeighborsClassifier # kNN
from sklearn.metrics import classification_report # kNN / LogReg
from sklearn.linear_model import LogisticRegression # LogReg
from keras.models import Sequential # nn
from keras.layers import Dense # nn

"""
Below are the functions that are used in the folder Benchmark too.
This is a copy but then with altered file paths and the k added to functions.
The output is rooted to this directory.
The validation data is anonymised and thus will not be imported from the data folder.
Please see preprocessing.py for the details.
"""

#####       KNN         #####
def measure_knn_adult(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Energy_knn.csv')

    label = 'Adult_' + str(k)

    with pyRAPL.Measurement(label, output=csv_output):
        # load data
        adult_train_data = pd.read_csv('output/processed/Adult_' + str(k) + '_train_data.csv')
        adult_train_labels = pd.read_csv('output/processed/Adult_' + str(k) + '_train_labels.csv')
        adult_val_data = pd.read_csv('output/processed/Adult_' + str(k) + '_val_data.csv')
        adult_val_labels = pd.read_csv('output/processed/Adult_' + str(k) + '_val_labels.csv')

        # make and fit kNN model
        NeighbourModel = KNeighborsClassifier(n_neighbors=5)
        NeighbourModel.fit(adult_train_data, adult_train_labels.values.ravel())

        # make report and add to dataframe
        report = classification_report(adult_val_labels, NeighbourModel.predict(adult_val_data), output_dict=True)

        # save data to csv file
        df = pd.DataFrame([report['accuracy']])
        df.columns = ['Accuracy']
        df.to_csv('Accuracies/knn_adult_' +str(k) + '.csv', index=False)

    # save energy consumption
    csv_output.save()

def measure_knn_student(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Energy_knn.csv')

    label = 'Student_' + str(k)
    with pyRAPL.Measurement(label, output=csv_output):
        # load data
        student_train_data = pd.read_csv('output/processed/Student_' + str(k) + '_train_data.csv')
        student_train_grade = pd.read_csv('output/processed/Student_' + str(k) + '_train_labels.csv')
        student_val_data = pd.read_csv('output/processed/Student_' + str(k) + '_val_data.csv')
        student_val_grade = pd.read_csv('output/processed/Student_' + str(k) + '_val_labels.csv')

        # make and fit kNN model for k = 5
        NeighbourModel = KNeighborsClassifier(n_neighbors=5)
        NeighbourModel.fit(student_train_data, student_train_grade.values.ravel())

        # make report and add to dataframe
        report = classification_report(student_val_grade, NeighbourModel.predict(student_val_data), output_dict=True)

        # save data to csv file
        df = pd.DataFrame([report['accuracy']])
        df.columns = ['Accuracy']
        df.to_csv('Accuracies/knn_student_' + str(k) + '.csv', index=False)

    # save energy consumption
    csv_output.save()

#####       Logistic Regression     #####
def measure_logreg_adult(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Energy_logreg.csv')

    label = 'Adult_' + str(k)

    with pyRAPL.Measurement(label, output=csv_output):
        # load data
        adult_train_data = pd.read_csv('output/processed/Adult_' + str(k) + '_train_data.csv')
        adult_train_labels = pd.read_csv('output/processed/Adult_' + str(k) + '_train_labels.csv')
        adult_val_data = pd.read_csv('output/processed/Adult_' + str(k) + '_val_data.csv')
        adult_val_labels = pd.read_csv('output/processed/Adult_' + str(k) + '_val_labels.csv')

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
        df.to_csv('Accuracies/logreg_adult_' +str(k) + '.csv', index=False)
    
    # save energy consumption
    csv_output.save()

def measure_logreg_student(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Energy_logreg.csv')

    label = 'Student_' + str(k)
    with pyRAPL.Measurement(label, output=csv_output):
        # load data
        student_train_data = pd.read_csv('output/processed/Student_' + str(k) + '_train_data.csv')
        student_train_grade = pd.read_csv('output/processed/Student_' + str(k) + '_train_labels.csv')
        student_val_data = pd.read_csv('output/processed/Student_' + str(k) + '_val_data.csv')
        student_val_grade = pd.read_csv('output/processed/Student_' + str(k) + '_val_labels.csv')

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
        df.to_csv('Accuracies/logreg_student_' +str(k) + '.csv', index=False)
    
    # save energy consumption
    csv_output.save()

#####       Logistic Regression     #####
def measure_nn_adult(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Energy_nn.csv')

    label = 'Adult_' + str(k)

    with pyRAPL.Measurement(label, output=csv_output):
        # load data
        adult_train_data = pd.read_csv('output/processed/Adult_' + str(k) + '_train_data.csv')
        adult_train_labels = pd.read_csv('output/processed/Adult_' + str(k) + '_train_labels.csv')
        adult_val_data = pd.read_csv('output/processed/Adult_' + str(k) + '_val_data.csv')
        adult_val_labels = pd.read_csv('output/processed/Adult_' + str(k) + '_val_labels.csv')

        # create neural network with keras sequential model
        NeuralNet = Sequential()

        # add 3 layers, one input, one output and one hidden layer
        NeuralNet.add(Dense(8, activation = 'relu'))
        NeuralNet.add(Dense(16, activation = 'relu'))
        NeuralNet.add(Dense(1, activation = 'sigmoid'))
        NeuralNet.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # train the network with training set and training labels
        # print("Training the neural network..\n")                   
        NeuralNet.fit(adult_train_data, adult_train_labels, batch_size = 150, epochs = 20, verbose=0)

        # validate the network with the validation set and labels
        # print("\nPrediction accuracy:\n")
        predication = NeuralNet.predict(adult_val_data)

        # print the accuracy
        loss, accuracy = NeuralNet.evaluate(adult_val_data, adult_val_labels, verbose=0)
        # print(f"\nThe loss is {loss}, and the accuracy is {accuracy}")

        # save results to csv
        df = pd.DataFrame([[accuracy]])
        df.columns = ['Accuracy']
        # note: we use mode append since the accuracy may differ a little per run
        df.to_csv('Accuracies/nn_adult_' +str(k) + '.csv', header=['Accuracy'], index=False, mode='a')
    
    # save energy consumption
    csv_output.save()

def measure_nn_student(k):
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('Energy/Energy_nn.csv')

    label = 'Student_' + str(k)
    with pyRAPL.Measurement(label, output=csv_output):
        # load data
        student_train_data = pd.read_csv('output/processed/Student_' + str(k) + '_train_data.csv')
        student_train_grade = pd.read_csv('output/processed/Student_' + str(k) + '_train_labels.csv')
        student_val_data = pd.read_csv('output/processed/Student_' + str(k) + '_val_data.csv')
        student_val_grade = pd.read_csv('output/processed/Student_' + str(k) + '_val_labels.csv')

        # create neural network with keras sequential model
        NeuralNet = Sequential()

        # add 3 layers, one input, one output and one hidden layer
        NeuralNet.add(Dense(8, activation = 'relu'))
        NeuralNet.add(Dense(16, activation = 'relu'))
        NeuralNet.add(Dense(1, activation = 'sigmoid'))
        NeuralNet.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # train the network with training set and training labels
        # print("Training the neural network..\n")
        NeuralNet.fit(student_train_data, student_train_grade, batch_size = 5, epochs = 20, verbose=0)

        # validate the network with the validation set and labels
        # print("\nPrediction accuracy:\n")
        predication = NeuralNet.predict(student_val_data)

        # print the accuracy
        loss, accuracy = NeuralNet.evaluate(student_val_data, student_val_grade, verbose=0)
        # print(f"\nThe loss is {loss}, and the accuracy is {accuracy}")

        # save results to csv
        df = pd.DataFrame([[accuracy]])
        df.columns = ['Accuracy']
        # note: we use mode append since the accuracy may differ a little per run
        df.to_csv('Accuracies/nn_student_' +str(k) + '.csv', header=['Accuracy'], index=False, mode='a')

    # save energy consumption
    csv_output.save()

if __name__ == '__main__':

    print("\nStarting with k-nearstest neighbours")

    ks = [3,10,27]
    for k in ks:
        for _ in range(1,11):
            measure_knn_adult(k)
            # report progress
            print(f"Adult {k}: Measurement {_} of 10 is saved to csv.")
        
        print('\n') # readability
        
        for _ in range(1,11):
            measure_knn_student(k)
            # report progress
            print(f"Student {k}: Measurement {_} of 10 is saved to csv.")

        print('\n') # readability
    
    print("\nStarting with logistic regression")
    for k in ks:
        for _ in range(1,11):
            measure_logreg_adult(k)
            # report progress
            print(f"Adult {k}: Measurement {_} of 10 is saved to csv.")

        print('\n') # readability

        for _ in range(1,11):
            measure_logreg_student(k)
            # report progress
            print(f"Student {k}: Measurement {_} of 10 is saved to csv.")

        print('\n') # readability

    print("\nStarting with neural network")
    # disable warnings from tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # disable copy warnings
    pd.set_option('mode.chained_assignment', None)

    for k in ks:
        for _ in range(1,11):
            measure_nn_adult(k)
            # report progress
            print(f"Adult {k}: Measurement {_} of 10 is saved to csv.")

        print('\n') # readability

        for _ in range(1,11):
            measure_nn_student(k)
            # report progress
            print(f"Student {k}: Measurement {_} of 10 is saved to csv.")
        print('\n') # readability
