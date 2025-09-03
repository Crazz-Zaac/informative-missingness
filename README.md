# Informative Missingness

### 📁 Project Structure
```bash
project/
├── run_exp_rf.py                # Main training script to train models
├── configs/
│   └── config.yml                 # Configuration for all the Machine Learning models
├── dataset/                       # Data loading & preprocessing
│   ├── preprocessed_tabular/     # Stores preprocessed tabular data
│   ├── raw/                      # Stores raw data before preprocessing
│   └── temp/                     # Stores intermediate data (from PostgreSQL)
├── db_utils/                     # Configurations for the PostgreSQL database
├── notebooks/                    # Jupyter notebooks for exploration and debugging
├── outputs/                      # output for all the experiments
│   └── experiments
│   │   ├── 20250811_222344         # This folder is created based on the date and time
│   │   │   └── logs/               # Stores logs for different ML models
│   │   │   └── models/             # logs models training parameters  
│   │   │   └── results/            # stores models results
├── plots/                        # Stores performance plots and visualizations
├── scripts/                      # Scripts to prepare raw data by querying PostgreSQL
│   └── fetch_demographics_data.py      # fetches the demographics data from DB and merge the target file
│   └── fetch_labevents_data.py         # fetches the labevents data from DB
│   └── prepare_data.py                 # main entry point to the raw data extraction process pipeline
├── src/
│   ├── config/
│   │   └── schemas.py            # Pydantic validation for classes, methods, and data types
│   ├── data/                     # Data handling and preprocessing
│   │   ├── data_loader.py        # Loads data from sources
│   │   ├── dataset.py            # Dataset logic and split handling
│   │   ├── data_preprocessing.py   # Data preprocessing methods definition
│   │   ├── tabular_data_preprocessor.py   # Tabular data preprocessing
│   │   └── temporal_preprocessing.py  # Temporal feature engineering
│   ├── models/
│   │   └── random_forest.py      # Random Forest model definition
│   │   └── gradient_boosting.py      # Gradient Boosting model definition
│   │   └── XGBoost.py      # XGBoost model definition
│   │   └── CatBoost.py      # CatBoost model definition
│   ├── training/
│   │   └── train_rf.py           # Training logic for Random Forest
│   │   └── train_gradboost.py           # Training logic for Gradient Boosting
│   │   └── XGBoost.py           # Training logic for XGBoost
│   │   └── CatBoost.py           # Training logic for CatBoost
│   └── utils/
│       └── logging_utils.py      # Logging configuration and setup
├── train_models.sbatch                # Script to assign model training process in HPC
```
---

#### Raw data extraction process pipeline
- For extracting and preparing the raw data, the `scripts/prepare_data.py` need to be executed. The script format is:
```shell
python scripts/prepare_data.py --cohort <cohort type (apl or nf)> --days <days (7 or 14)> --cohort_target <cohort_name>

# Example:
python scripts/prepare_data.py --cohort apl --days 14 --cohort_target "mimic_iv_target_apl_tr_14_days.csv.gz"
```
- The `cohort_name` is the name of the file in the `MIMIC-IV-data/` folder. In our case:
    - `mimic_iv_target_apl_tr_14_days.csv.gz`
    - `mimic_iv_target_nf_14_days.csv.gz`

- This will create a `apl_lab_events_data_with_demographics.parquet` in `raw/apl_lab_events_data_with_demographics.parquet` which will be used during preprocessing.

---

#### Running models in docker container
- Every time a new package is added or a change is made to the project, it is necessary to build the image. This will create a `model-training` container with all the necessary packages. 
```bash
docker compose build --no-cache             # build the image
```
- Once the container is started, one or more models can be trained on independent containers separated by spaces.  
```bash
docker compose up randomforest     # will start randomforest-trainer container 
docker compose up randomforest xgboost catboost   # will start randomforest-trainer xgboost-trainer and catboost-trainer container independently
```


#### Running the pipeline
```python
cd project
python run_exp.py
```


#### Running the pipeline in `HPC` server
- Clone the repo
```bash
cd informative-missingness
sbatch -p work train_models.sbatch
```
- Copying raw data files from local to remote
```bash
scp -r raw/aplasia_*.parquet csnhr.nhr.fau.de:informative-missingness/project/dataset/raw/
```

##### Queries to load the `mimiciv` data
- The `.sql` files in `/postgres` is copied from the [MIMIC Github Repository](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) 

- Model Calibration with [PyCalEva](https://martinweigl.github.io/pycaleva/)