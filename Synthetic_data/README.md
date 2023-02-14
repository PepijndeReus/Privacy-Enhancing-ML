# Synthetic data
This folder contains the files to create a synthetic data set using the [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer).

## Instructions for reproduction
To create the synthetic data, one should first run the [cleaning](https://github.com/PepijndeReus/ThesisAI/blob/main/Data/Cleaning.py) and [preprocessing](https://github.com/PepijndeReus/ThesisAI/blob/main/Data/Preprocessing.py) files in the [Data folder](https://github.com/PepijndeReus/ThesisAI/blob/main/Data).

1. Run create_set.py script to create synthetic data
2. Preprocess the data using preprocessing.py
3. Run the results using run_results_synt.py
Then first run the create_set.py script to create synthetic data sets for both the adult and student performance data set. The results will be synthetic data sets, saved as comma-separated values files (.csv).
These data sets should then be processed using the preprocessing Python script.
Afterwards, run_results_synt.py will use the preprocessed data sets to train the models and output the performance and accuracy.

The summary of results of this code can be found in [this notebook](https://github.com/PepijndeReus/ThesisAI/blob/main/Synthetic_data/_Analysis.ipynb).
Please note that preprocessing is required twice, once to remove empty values from the raw data and then again to transform the data into training and validation data.