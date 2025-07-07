# Informative Missingness

### 📁 Project Structure
```bash
project/
├── run_train_rf.py                # Main training script to train Random Forest model
├── configs/
│   └── config.yml                 # Configuration for all the Machine Learning models
├── dataset/                       # Data loading & preprocessing
│   ├── preprocessed_tabular/     # Stores preprocessed tabular data
│   ├── raw/                      # Stores raw data before preprocessing
│   └── temp/                     # Stores intermediate data (from PostgreSQL)
├── db_utils/                     # Configurations for the PostgreSQL database
├── logs/                         # Stores logs for different ML models
├── notebooks/                    # Jupyter notebooks for exploration and debugging
├── plots/                        # Stores performance plots and visualizations
├── scripts/                      # Scripts to prepare raw data by querying PostgreSQL
│   └── fetch_labevents_data.py  # Script to fetch labevents data from DB
├── src/
│   ├── config/
│   │   └── schemas.py            # Pydantic validation for classes, methods, and data types
│   ├── data/                     # Data handling and preprocessing
│   │   ├── data_loader.py        # Loads data from sources
│   │   ├── dataset.py            # Dataset logic and split handling
│   │   ├── tabular_preprocessing.py   # Tabular preprocessing methods
│   │   └── temporal_preprocessing.py  # Temporal feature engineering
│   ├── model/
│   │   └── random_forest.py      # Random Forest model definition
│   ├── training/
│   │   └── train_rf.py           # Training logic for Random Forest
│   └── utils/
│       └── logging_utils.py      # Logging configuration and setup
```

#### Running the pipeline
```python
cd project
python run_train_rf.py
```


#### Running the pipeline in `HPC` server
- Clone the repo
```bash
cd informative-missingness
sbatch -p work train_rf.sbatch
```


##### Queries to load the `mimiciv` data
- The `.sql` files in `/postgres` is copied from the [MIMIC Github Repository](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) 

- Model Calibration with [PyCalEva](https://martinweigl.github.io/pycaleva/)