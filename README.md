# Informative Missingness

### 📁 Project Structure
```bash
project/
├── run_exp_rf.py                 # Main training script to train models
├── configs/
│   └── config.yml                # Configuration for all the Machine Learning models
├── dataset/                      # Data loading & preprocessing
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
├── plots/                          # Stores performance plots and visualizations
├── scripts/                        # Scripts to prepare raw data by querying PostgreSQL
│   └── fetch_demographics_data.py      # fetches the demographics data from DB and merge the target file
│   └── fetch_labevents_data.py         # fetches the labevents data from DB
│   └── prepare_data.py                 # main entry point to the raw data extraction process pipeline
├── src/
│   ├── config/
│   │   └── schemas.py            # Pydantic validation for classes, methods, and data types
│   ├── data/                     # Data handling and preprocessing
│   │   ├── data_loader.py        # Loads data from sources
│   │   ├── dataset.py            # Dataset logic and split handling
│   │   ├── data_preprocessing.py               # Data preprocessing methods definition
│   │   ├── tabular_data_preprocessor.py        # Tabular data preprocessing
│   │   └── temporal_preprocessing.py           # Temporal feature engineering
│   ├── models/
│   │   └── random_forest.py        # Random Forest model definition
│   │   └── gradient_boosting.py    # Gradient Boosting model definition
│   │   └── XGBoost.py              # XGBoost model definition
│   │   └── CatBoost.py             # CatBoost model definition
│   ├── training/
│   │   └── train_rf.py             # Training logic for Random Forest
│   │   └── train_gradboost.py      # Training logic for Gradient Boosting
│   │   └── XGBoost.py              # Training logic for XGBoost
│   │   └── CatBoost.py             # Training logic for CatBoost
│   └── utils/
│       └── logging_utils.py        # Logging configuration and setup
├── train_models.sbatch             # Slurm script to assign model training process in HPC
├── run_exp.py                      # this script is only used while training models without docker
├── train_model.py                  # this script is used while training models in docker container
docker/
├── Dockerfile                      # All the necessary configurations during container creation
docker-compose.yml                  # Docker container build related configuration
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

#### Running the pipeline locally
- Every time a *new package* is added or a change is made to the project, it is necessary to re-build the image. This will create a `model-training` container with including all the packages. 
1. Using docker
```bash
docker compose build --no-cache             # builds the image
```
- Once the container is started, one or more models can be trained on independent containers separated by spaces.  
```bash
docker compose up randomforest                      # starts randomforest-trainer container 
docker compose up randomforest xgboost catboost     # starts randomforest-trainer xgboost-trainer and catboost-trainer container independently
```

2. Without using docker

```bash 
cd project
python run_exp.py
```


#### Running the pipeline in `HPC` server

1. Using `apptainer/docker`
- Create wheels (`.whl`) files to avoid package dependency conflicts
```bash
pip download -r requirements.txt -d wheels/
```

- Build the container (`.tar`) file
```bash
docker compose build --no-cache 
```

- Copy the `.tar` file to HPC
```bash
scp model-training.tar USERNAME@CLUSTER:/home/USERNAME/project_folder/model-training.tar  
```

- Creating `.sif` file
```bash
apptainer build model-training.sif docker-archive://model-training.tar          # create .sif file 
```

- Run the slurm job
```bash
sbatch -p work train_models.sbatch
```


2. Without using `apptainer/docker`
Uncomment the line:
```bash
python project/run_exp.py
```
and comment out these lines:
```bash
models=("RandomForest" "GradientBoosting" "LogisticRegression" "XGBoost" "CatBoost")
MODEL=${models[$SLURM_ARRAY_TASK_ID]}

echo "Training model: $MODEL"

apptainer exec --nv -B $PWD:/project model-training.sif \
    python project/train_model.py --model $MODEL
```

---

#### Helpful commands 
- Copying raw data files from local to remote
```bash
scp -r raw/apl_*.parquet USERNAME@CLUSTER:/home/USERNAME/
```

##### Queries to load the `mimiciv` data
- The `.sql` files in `/postgres` is copied from the [MIMIC Github Repository](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) 

- Model Calibration with [PyCalEva](https://martinweigl.github.io/pycaleva/)