"""
Code that runs a neutral network on the adult and student performance dataset without annoymisation.
Supposed to be a benchmark for the anonimised data sets.
"""
# import libraries
import pandas as pd
import numpy as np
import keras
import os
import pyRAPL

# import modules
from keras.models import Sequential
from keras.layers import Dense

# disable copy warnings
pd.set_option('mode.chained_assignment', None)

def measure_nn_adult():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../../Benchmark/Performance/Energy_nn.csv')

    with pyRAPL.Measurement('Adult', output=csv_output):
        # load data
        adult_train_data = pd.read_csv('Adult_train_data.csv')
        adult_train_labels = pd.read_csv('Adult_train_labels.csv')
        adult_val_data = pd.read_csv('Adult_val_data.csv')
        adult_val_labels = pd.read_csv('Adult_val_labels.csv')

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
        df.to_csv('../../Benchmark/Performance/Accuracy_nn_adult.csv', header=['Accuracy'], index=False, mode='a')
    
    # save energy consumption
    csv_output.save()

def measure_nn_student():
    # measure energy consumption
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('../../Benchmark/Performance/Energy_nn.csv')

    with pyRAPL.Measurement('Student', output=csv_output):
        # load data
        student_train_data = pd.read_csv('student_train_data.csv')
        student_train_grade = pd.read_csv('student_train_grade.csv')
        student_val_data = pd.read_csv('student_val_data.csv')
        student_val_grade = pd.read_csv('student_val_grade.csv')

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
        df.to_csv('../../Benchmark/Performance/Accuracy_nn_student.csv', header=['Accuracy'], index=False, mode='a')

    # save energy consumption
    csv_output.save()

if __name__ == '__main__':
    # change directory for Adult data set
    os.chdir('../Data/Adult')
    for _ in range(1,11):
        measure_nn_adult()
        # report progress
        print(f"Adult: Measurement {_} of 10 is saved to csv.")
    
    # change directory for Student data set
    os.chdir('../Student')
    for _ in range(1,11):
        measure_nn_student()
        # report progress
        print(f"Student: Measurement {_} of 10 is saved to csv.")