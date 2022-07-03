This folder contains the files to create a synthetic data set using the [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer).
To create the synthetic data, one should first run the [cleaning](https://github.com/PepijndeReus/ThesisAI/blob/main/Data/Cleaning.py) and [preprocessing](https://github.com/PepijndeReus/ThesisAI/blob/main/Data/Preprocessing.py) files in the [Data folder](https://github.com/PepijndeReus/ThesisAI/blob/main/Data).

Then first run the create_set.py script to create synthetic data sets for both the adult and student performance data set. The results will be synthetic data sets, saved as comma-separated values files (.csv).
These data sets should then be processed using the preprocessing Python script.
Afterwards, run_results_synt.py will use the preprocessed data sets to train the models and output the performance and accuracy.

The summary of results of this code can be found in [this file](https://github.com/PepijndeReus/ThesisAI/blob/main/Synthetic_data/_Analysis.ipynb).
