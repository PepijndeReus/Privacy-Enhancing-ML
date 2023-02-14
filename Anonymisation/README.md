# Anonymisation
This folder contains the files needed for the anonymisation of the data sets.
The code is written in a combination of Python and Java, which makes the use of a [library](https://github.com/PepijndeReus/ThesisAI/tree/main/Anonymisation/libraries) folder necessary.
Please note that the [hierarchy](https://github.com/PepijndeReus/ThesisAI/tree/main/Anonymisation/hierarchy) folder and it's components are to be made manually for each data set. Hierarchy links to adult and [hierarchy_student](https://github.com/PepijndeReus/ThesisAI/tree/main/Anonymisation/hierarchy_student) links to hierarchy for the student data set.

## Instructions for reproduction
To run the anonymisation code, use the following instructions:
1. Make sure to use the right hierarchy (see Hierachy section below)
2. First compile the code using:
```
javac -cp .:libraries/* k_anonymity.java
```
3. Then run the code using
```
python3 k-anonymity.py {dataset}.yaml
```
where dataset is to be replaced by either student or adult.
4. Preprocess the data using preprocessing.py
5. Run the results using run_results_anony.py
The output will now be stored in the folders Accuracies, Energy and output.

The summary of results of this code can be found in [this notebook](https://github.com/PepijndeReus/ThesisAI/blob/main/Anonymisation/_Analysis.ipynb).

### Hierarchy
Copy and paste the right hierarchy into the hierarchy.txt file before running the experiment. Due to time contraints this, unfortunately, has not been automised.

**Adult hierarchy** :\
age,Quasi_identifying,Decimal,arithmic_mean
type_employer,Quasi_identifying,Decimal,arithmic_mean
education,Quasi_identifying,Decimal,arithmic_mean
marital,Quasi_identifying,Decimal,arithmic_mean
occupation,Quasi_identifying,Decimal,arithmic_mean
relationship,Quasi_identifying,Decimal,arithmic_mean
race,Quasi_identifying,Decimal,arithmic_mean
sex,Quasi_identifying,NaN,arithmic_mean
capital_gain,Quasi_identifying,Decimal,arithmic_mean
capital_loss,Quasi_identifying,Decimal,arithmic_mean
hr_per_week,Quasi_identifying,Decimal,arithmic_mean
country,Quasi_identifying,Decimal,arithmic_mean
income,Insensitive,NaN,NaN

**Student hierarchy**:\
school,Quasi_identifying,Decimal,arithmic_mean
sex,Quasi_identifying,Decimal,arithmic_mean
age,Quasi_identifying,Decimal,arithmic_mean
address,Quasi_identifying,Decimal,arithmic_mean
famsize,Quasi_identifying,Decimal,arithmic_mean
Pstatus,Quasi_identifying,Decimal,arithmic_mean
Medu,Quasi_identifying,Decimal,arithmic_mean
Fedu,Quasi_identifying,Decimal,arithmic_mean
Mjob,Quasi_identifying,Decimal,arithmic_mean
Fjob,Quasi_identifying,Decimal,arithmic_mean
reason,Quasi_identifying,Decimal,arithmic_mean
guardian,Quasi_identifying,Decimal,arithmic_mean
traveltime,Quasi_identifying,Decimal,arithmic_mean
studytime,Quasi_identifying,Decimal,arithmic_mean
failures,Quasi_identifying,Decimal,arithmic_mean
schoolsup,Quasi_identifying,Decimal,arithmic_mean
famsup,Quasi_identifying,Decimal,arithmic_mean
paid,Quasi_identifying,Decimal,arithmic_mean
activities,Quasi_identifying,Decimal,arithmic_mean
nursery,Quasi_identifying,Decimal,arithmic_mean
higher,Quasi_identifying,Decimal,arithmic_mean
internet,Quasi_identifying,Decimal,arithmic_mean
romantic,Quasi_identifying,Decimal,arithmic_mean
famrel,Quasi_identifying,Decimal,arithmic_mean
freetime,Quasi_identifying,Decimal,arithmic_mean
goout,Quasi_identifying,Decimal,arithmic_mean
Dalc,Quasi_identifying,Decimal,arithmic_mean
Walc,Quasi_identifying,Decimal,arithmic_mean
health,Quasi_identifying,Decimal,arithmic_mean
absences,Quasi_identifying,Decimal,arithmic_mean
G1,Quasi_identifying,Decimal,arithmic_mean
G2,Quasi_identifying,Decimal,arithmic_mean
G3,Insensitive,NaN,NaN
