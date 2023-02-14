# ThesisAI
This is the repository that contains the code for the paper (*available after acceptance*). The code may be reproduced by referring to this paper. All files should have sufficient documentation for reproduction and understanding. Any remaining questions or comments may be sent to *e-mail available after review*.
**Please note** that some links to folders and/or files may be invalid due to the anonymisation, these links will be working for the camera-ready paper. Nevertheless the links are provided for simplicity and the words used in the links correspond with their respective files and folders.

## About the repository
The repository is structured with the following folders and files:
### Data
This folder contains the data sets obtained from the UCI machine learning repository, separated using two different folders. It also contains two Python files to clean and preprocess the data sets as described in the Experimental Setup of the paper. After running these files the Energy folder will be available containing the energy consumptions of the data preprocessing and cleaning.

### Benchmark
The benchmark folder contains the Python scripts for three different machine learning models and one script (run_results.py) that combines these three models to obtain results. After running the results the Performance folder will be filled with measurements of the accuracy and energy consumption for this benchmark.

### Anonymisation and Synthetic data
The folders for [Anonymisation](https://github.com/PepijndeReus/ThesisAI/tree/main/Anonymisation) and [Synthetic data](https://github.com/PepijndeReus/ThesisAI/tree/main/Synthetic_data) have separate readme files with introduction and instructions to the code.

## Notebooks
Three notebooks are provided that use the results to summarise and visualise the data from the experiments. [This notebook](https://github.com/PepijndeReus/ThesisAI/blob/main/_Analysis.ipynb) links to the energy consumptions provided in Tables II-IV and VI. [This notebook](https://github.com/PepijndeReus/ThesisAI/blob/main/analysis_paper.ipynb) contains the plots used in Figures 3 and 4 of the paper. Finally [this notebook](https://github.com/PepijndeReus/ThesisAI/blob/main/MannWhitney.ipynb) contains the code required for the Mann Whitney U test presented in Table V.

## Other
The gitignore file is set up to ignore preprocessed data sets and results to keep the repository small in size. It also ignores .ipynb files and the checkpoints of these as these were used for development purposes only.
