"""
Creating the synthetic data using the Git page from DataSynthesizer
https://github.com/DataResponsibly/DataSynthesizer/blob/master/notebooks/DataSynthesizer__correlated_attribute_mode.ipynb

"""

import pyRAPL

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
# unused: from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

def create_data_adult():
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('./Energy/creating_synthetic_adult.csv')

    with pyRAPL.Measurement('Adult', output=csv_output):
        # input dataset
        input_data = '../Data/Adult/Adult_train.csv'

        # location of two output files
        mode = 'correlated_attribute_mode'
        description_file = f'./output/description_adult.json'
        synthetic_data = f'./output/adult_synthetic_data.csv'

        # An attribute is categorical if its domain size is less than this threshold.
        # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
        threshold_value = 42

        # Specify categorical attributes
        # categorical_attributes = {'type_employer': True, 'education': True, 'marital': True, 'occupation': True, 'relationship': True, 
        #                         'race': True, 'sex': True, 'country': True, 'income': True}

        # A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not 
        # change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
        # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
        epsilon = 0

        # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
        degree_of_bayesian_network = 2

        # Number of tuples generated in synthetic dataset.
        num_tuples_to_generate = 30163

        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                                epsilon=epsilon, 
                                                                k=degree_of_bayesian_network)
        describer.save_dataset_description_to_file(description_file)

        # Generate data set
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
        generator.save_synthetic_data(synthetic_data)
    csv_output.save()

def create_data_student():
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput('./Energy/creating_synthetic_student.csv')

    with pyRAPL.Measurement('Student', output=csv_output):
        # input dataset
        input_data = '../Data/Student/student-por.csv'

        # location of two output files
        mode = 'correlated_attribute_mode'
        description_file = f'./output/description_student.json'
        synthetic_data = f'./output/student_synthetic_data.csv'

        # An attribute is categorical if its domain size is less than this threshold.
        # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
        threshold_value = 20

        # A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not 
        # change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
        # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
        epsilon = 0

        # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
        degree_of_bayesian_network = 2

        # Number of tuples generated in synthetic dataset.
        num_tuples_to_generate = 649

        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                                epsilon=epsilon, 
                                                                k=degree_of_bayesian_network)
        describer.save_dataset_description_to_file(description_file)

        # Generate data set
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
        generator.save_synthetic_data(synthetic_data)
    csv_output.save()

if __name__ == '__main__':
    # measure idle energy consumption
    for _ in range(1,11):
        create_data_adult()
        print(f"Adult: Synthetic measurement {_} of 10 is saved to csv.")

    for _ in range(1,11):
        create_data_student()
        print(f"Student: Synthetic measurement {_} of 10 is saved to csv.")