## Purpose of this directory 

- This folder is intended to store the **initial cohort file**, which is required to extract data from the database. Create a directory named `dataset/MIMIC-IV-data/ `and place the cohort files here.

- When the scripts in the scripts/ directory are executed, two subdirectories — `dataset/raw` and `dataset/temp` — will be created automatically to hold intermediate files.

- During model training, another directory `dataset/preprocessed_tabular` will be generated. It stores the preprocessed data prior to model input, allowing for quick inspection of the prepared datasets.