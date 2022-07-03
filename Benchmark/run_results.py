import K_nearest
import LogisticRegression
import NeuralNetwork
import os
import time
import pyRAPL

if __name__ == '__main__':
    # measure idle energy consumption
    for _ in range(1,11):
        pyRAPL.setup()
        csv_output = pyRAPL.outputs.CSVOutput('Performance/Energy_idle.csv')
        with pyRAPL.Measurement('Idle', output=csv_output):
            time.sleep(1)
        csv_output.save()

    # disable warnings from tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # change directory for Adult data set
    os.chdir('../Data/Adult')
    print("\nStarting with k-nearstest neighbours")
    for _ in range(1,11):
        K_nearest.measure_knn_adult()
        # report progress
        print(f"Adult: Measurement {_} of 10 is saved to csv.")
    
    # change directory for Student data set
    os.chdir('../Student')
    for _ in range(1,11):
        K_nearest.measure_knn_student()
        # report progress
        print(f"Student: Measurement {_} of 10 is saved to csv.")
    
    print("\nStarting with logistic regression")
    # change directory for Adult data set
    os.chdir('../Adult')
    for _ in range(1,11):
        LogisticRegression.measure_logreg_adult()
        # report progress
        print(f"Adult: Measurement {_} of 10 is saved to csv.")
    
    # change directory for Student data set
    os.chdir('../Student')
    for _ in range(1,11):
        LogisticRegression.measure_logreg_student()
        # report progress
        print(f"Student: Measurement {_} of 10 is saved to csv.")
    
    print("\nStarting with neural network")
    # change directory for Adult data set
    os.chdir('../Adult')
    for _ in range(1,11):
        NeuralNetwork.measure_nn_adult()
        # report progress
        print(f"Adult: Measurement {_} of 10 is saved to csv.")
    
    # change directory for Student data set
    os.chdir('../Student')
    for _ in range(1,11):
        NeuralNetwork.measure_nn_student()
        # report progress
        print(f"Student: Measurement {_} of 10 is saved to csv.")