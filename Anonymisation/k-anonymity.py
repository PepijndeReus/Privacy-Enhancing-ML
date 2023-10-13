"""
This file is used to measure the energy consumption of running the ARX anonymisation algorithm.
Please note that the code was based upon earlier work by
https://github.com/ana-oprescu/Energy-cost-of-k-anonymity/blob/main/Anonymization/RAPL_on_Java.py
"""

import pyRAPL
import subprocess
import os.path
from subprocess import STDOUT, PIPE
import sys
import yaml
from yaml.loader import SafeLoader
import time

def Load_control_file(Test_file):
    if not os.path.isfile(Test_file):
        print("Given file does not exsist")
        exit()

    if not ".yaml" in Test_file:
        print("Control-file has to be a yaml file")
        exit()

    with open(Test_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
    
    return data

def create_gen(data):
    print("Create gen")
    f = open("hierarchy.txt", "w")
    for key, value in data["hierarchy"].items():
        f.write(key + "," + value + "," + "\n")
    f.close()

def create_hierachy(data):
    print("Create hierarchy")
    if data["type"] == "general":
        create_gen(data)
    else:
        print("Invalid type given")
        exit()

def measurement(k, iterations, input_file, class_name, suppression, data_type, dataset): 
    pyRAPL.setup()

    csv_output = pyRAPL.outputs.CSVOutput("Energy/Energy_anony_"+ str(k) + "_" + str(dataset) + ".csv")

    # delete old synthetic data if present
    if os.path.isfile(str(dataset)+'_'+ str(k) + '.csv'):
        os.remove(str(dataset)+'_'+ str(k) + '.csv')

    @pyRAPL.measureit(output=csv_output, number=1)
    def Energy_consumption():

        subprocess.run(['sudo', 'java', '-cp', '.:libraries/*', class_name, 
                        str(k), input_file, str(suppression), dataset], capture_output=False)
        
    for _ in range(iterations + 1):
        Energy_consumption()
        print(f"Measurement { _ + 1} out of {iterations} completed.")
    
    csv_output.save()
        
def idle_measurements():
    pyRAPL.setup()

    csv_output2 = pyRAPL.outputs.CSVOutput('Energy/Energy_idle.csv')
    
    @pyRAPL.measureit(output=csv_output2, number=10)
    def wait():
        time.sleep(1)

    wait()
    csv_output2.save()

    
if __name__ == '__main__':
    """
    Please note: compile Java file before running Python using:
    'sudo javac -cp .:libraries/* k_anonymity.java'
    arg1 (required): .yaml file for the data set
    
    """
    # print("Do not forget to compile first: 'javac -cp .:libraries/* k_anonymity.java'")

    # use YAML file as input
    Test_file = sys.argv[1]
    data = Load_control_file(Test_file)

    # create hierachy for the 
    create_hierachy(data)
    
    
    for k in data["k_values"]:
        measurement(k, int(data["iterations"]), data["input_file"], data["class_name"], 
                        data["suppression_limit"], data["type"], data["data_set"])
    
    # idle_measurements()
    print("Measure")
