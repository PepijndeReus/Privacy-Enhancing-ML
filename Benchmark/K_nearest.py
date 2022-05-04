"""
This file uses pyRAPL to measure the energyconsumption of the k-nearest neighbours machine learning model on both the adult and student data set.
"""

# import modules
import pandas as pd
import numpy as np
import os
import pyRAPL
# import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# measure energy consumption
pyRAPL.setup()
meter = pyRAPL.Measurement('kNN')
meter.begin()

# change directory for Adult data set
print("We zijn nu hier:")
print(os.getcwd())
os.chdir('../Data/Adult')

# load data
adult_train_data = pd.read_csv('Adult_train_data.csv')
adult_train_labels = pd.read_csv('Adult_train_labels.csv')
adult_val_data = pd.read_csv('Adult_val_data.csv')
adult_val_labels = pd.read_csv('Adult_val_labels.csv')

# list with k's
list_k = [2,3,5,7,10]

# make kNN for each k value in list
for k in list_k:
    # make and fit kNN model
    NeighbourModel = KNeighborsClassifier(n_neighbors=k)
    NeighbourModel.fit(adult_train_data, adult_train_labels.values.ravel())

    # make and print report
    report = classification_report(adult_val_labels, NeighbourModel.predict(adult_val_data), output_dict=True)
    print(f"For k = {k}, accuracy =", end=" ")
    print(report['accuracy'])

meter.end()
meter.export('knn_adult_energy.csv')