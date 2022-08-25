# ThesisAI
This is the repository that contains my bachelor thesis for the [Artificial Intelligence](https://www.uva.nl/programmas/bachelors/kunstmatige-intelligentie/kunstmatige-intelligentie.html) program of the University of Amsterdam. The thesis lead to [this paper](https://dspace.uba.uva.nl/server/api/core/bitstreams/66b80ca2-d130-40d2-a2ac-ef039f6b606c/content). The code may be reproduced by referring to the paper.

## About the repository
### Data
This folder contains the data sets obtained from the UCI machine learning repository, separated using two different folders.
The cleaning and preprocessing scripts have to be executed in order to use the data for the baseline/anonymisation/synthetic data. 

### Benchmark
The benchmark folder contains the Python scripts for three different machine learning models and one script (run_results.py) that combines these three models to obtain results. 

### Anonymisation and Synthetic data
The folders for [Anonymisation](https://github.com/PepijndeReus/ThesisAI/tree/main/Anonymisation) and [Synthetic data](https://github.com/PepijndeReus/ThesisAI/tree/main/Synthetic_data) have separate readme files with introduction to the code.

## Other
The gitignore file is set up to ignore processed data sets and results to keep the repository small in size. It also ignores .ipynb files and the checkpoints of these as these were used for development purposes only.
