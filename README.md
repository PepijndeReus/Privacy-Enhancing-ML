# ICT4S'23: Energy cost and machine learning accuracy impact of k-anonymisation and synthetic data techniques
This is the repository that contains the code for the [ICT4S'23 paper](https://ieeexplore.ieee.org/document/10292174). The code may be reproduced by referring to this paper. The presentation of the paper was recorded, you can find the recording [here](https://www.youtube.com/watch?v=WccsQ_PoL5U&t=3665s&ab_channel=ICT4SConferenceOnlineChannel). The [slides](https://github.com/PepijndeReus/Privacy-Enhancing-ML/blob/main/ICT4S23-presentation.pptx) are provided in this repository. All files should have sufficient documentation for reproduction and understanding. Any remaining questions or comments may be sent to [my e-mail](mailto:p.dereus@uva.nl).

The article and/or this repository should be cited as:
```
@inproceedings{de2023energy,
  title={Energy cost and machine learning accuracy impact of k-anonymisation and synthetic data techniques},
  author={{de Reus}, Pepijn and Oprescu, Ana and {van Elsen}, Koen},
  booktitle={ICT4S, June 2023, Rennes, France},
  pages={57--65},
  year={2023},
  organization={International Conference on ICT for Sustainability}
}
```

## About this repository
The repository is structured with the following folders and files:
### Data
This folder contains the data sets obtained from the UCI machine learning repository, separated using two different folders. It also contains two Python files to clean and preprocess the data sets as described in the Experimental Setup of the paper. After running these files the Energy folder will be available containing the energy consumptions of the data preprocessing and cleaning.

### Benchmark
The benchmark folder contains the Python scripts for three different machine learning models and one script (run_results.py) that combines these three models to obtain results. After running the results the Performance folder will be filled with measurements of the accuracy and energy consumption for this benchmark.

### Anonymisation and Synthetic data
The folders for [Anonymisation](https://github.com/PepijndeReus/Privacy-Enhancing-ML/tree/main/Anonymisation) and [Synthetic data](https://github.com/PepijndeReus/Privacy-Enhancing-ML/tree/main/Synthetic_data) have separate readme files with introduction and instructions to the code. The folders are used for anonymising or synthesising the data and performing the experiments, after which the results will be stored in these folders respectively. The hyperparameters used in our paper are included in the synthetic data generation code and anonymisation code.

### Notebooks
Two notebooks are provided that use the results to summarise and visualise the data from the experiments. [This notebook](https://github.com/PepijndeReus/Privacy-Enhancing-ML/blob/main/analysis_paper.ipynb) contains the plots used in Figures 3 and 4 of the paper. Finally [this notebook](https://github.com/PepijndeReus/Privacy-Enhancing-ML/blob/main/MannWhitney.ipynb) contains the code required for the Mann Whitney U test presented in Table V.

The used notebooks for Tables II-IV and VI are available in the folders Anonymisation and Synthetic data.

### Miscellaneous
The gitignore file is set up to ignore preprocessed data sets and results to keep the repository small in size. It also ignores .ipynb files and the checkpoints of these as these were used for development purposes only.
