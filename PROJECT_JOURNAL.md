## P10 - Informative Missingness Project Journal

### 2025-05-09
- [x] Attended a general meeting (10 AM)
    - Key discussions:
        - Google Scholar / Scopus for finding relevant papers
        - FAU VPN 
        - creating virtual environments
        - LaTeX 
        - HPC training
        - Documenting and presenting work
- [x] Initialized the project structure and repository.
- [x] Set up `.ssh` keys for GitHub access.
- [x] Decided to log progress in the `PROJECT_JOURNAL.md`.
- [x] Created HPC account and accepted the invitation
- [x] Initially I had downloaded incomplete data, so had to download it again

---

### 2025-05-10: Data Preparation and Setup
- [x] Decided to use the docker version of `postgres` 
- [x] Configured the `docker-compose.yml` file with:
    - initial mount of `postgres/create.sql` and `postgres/load.sql`
- [x] The `create.sql` worked smoothly upon `docker-compose.yml` 
- [x] For loading the data I had to do it manually using the scripts below:
    - Copy `postgres/load.sql` to `load_mimic.sql`
        - `docker cp postgres/load.sql mimiciv_postgres:load_mimic.sql`
    - Then docker execute the `load_mimic.sql`
        - `docker exec -it mimiciv_postgres psql -U postgres -d mimiciv -f /load_mimic.sql`
    - After waiting for sometime, had to test it. So the following query is used to login to postgres and displaying the data from the `mimiciv_hosp.admissions` table.
        - `docker exec mimiciv_postgres psql -U postgres -d mimiciv -c "SELECT * FROM mimiciv_hosp.admissions;"`
    - Loading the `MIMIC-IV` data into PostgreSQL failed and terminated  because of millions of data. I had to increase the memory size in the [docker-compose.yml](./docker-compose.yml) file.
    ```bash
    psql:/load_mimic.sql:51: server closed the connection unexpectedly
            This probably means the server terminated abnormally
            before or while processing the request.
    FATAL:  terminating connection due to administrator command
    CONTEXT:  COPY inputevents, line 2912337: "12634755,25987676,36738588,..."
    psql:/load_mimic.sql:51: invalid socket
    psql:/load_mimic.sql:51: error: connection to server was lost

    ```
    - I had to increase the memory size and increase the number of CPUs to handle the overloading.
    ```bash
    deploy:
        resources:
            limits:
                cpus: '12'
                memory: 4G
            reservations:
                memory: 4G
    ```
    
### 2025-05-11
- [x] Created `utils/config.py` to store database settings
- [x] Created `utils/database.py` to:
    - connect to database
    - show list of all tables in schema
    - read a table table into a dataframe with limit
- [x] Made it possible to import into notebook using the command `pip install -e .`
- [x] Created a `notebooks/exp_2025.ipynb` for teseting the package

### 2025-05-16
- [x] Attended a meeting (11 AM)
    - The following things were discussed:
        - Purpose of research: Is missingness pattern predictive?
        - Finding what patterns have been investigated.
        - Possibility:
            - Different population have same missingness
            - Same population have same missingness
        - Preparation of data in two ways:
            - All raw data
            - Binarized version of data (e.g., if missing `0` else `1`)
        - Data can be organized in tabular or graph
        - Try to have more graphical representation of data
        - TO DOs:
            - Explore graphs over time
            - Explore patterns co-occurrence (i.e., how often the given variable occurs) -> `is it predictable?`
            - Check missingness across different demographic group
        - Model training:
            - Train temporal graph neural network (if possible)
            - Ask: How much information do I have to have to actually have a reasonable/good accuracy?
            - Look into the most discriminative patterns specific to a case (e.g., cancer or chemotherapy)
            - Train on `raw data`
            - Train on `indicators`
            - Training on both `raw` and `indicators`
            - Analyse the performance
        - Additional:
            - `Conformal Prediction` can also be experimented
        
- [x] Related Papers:
    1. [Modeling Missing Data in Clinical Time Series with RNNs](./papers/supervisors_recommendation/Lipton16.pdf)
    - Brief summary:
        - Imputation techniques
            - **Zero Imputation**: If a value $x^{t}$ is missing, then replace it with `0`
            - **Forward-Filling**:
                - If a previous value exists for a variable, it is carried forward. If there's no previous record measurement, it is imputed with the median estimated over all measurements in the training data of that variable.
            - **Missingness Indicators**: 
                - Binary variables are added to indicate if value was imputed. For each variable $x^{t}$, an indicator $m_i^{(t)}$ = 1 if missing otherwise 0.
            - **Hand-Engineered Missingness Features (for linear models)**:
                - Frequency of measurement 
                - Standard deviation of missingness
                - Whether a variable was measured at all
                - Timing of first/last measurement
        - Best model
            - LSTM (RNN):
                - Zero Imputation
                - Missing data indicators
        - Findings
            - RNNs fits better and can model complex dependencies.
            - Linear models also improve but are limited since only static weights can be assigned to missingness
    2. [Recurrent Neural Networks for Multivariate Time Series with Missing Values](./papers/supervisors_recommendation/s41598-018-24271-9.pdf)
    - Brief summary:
        - Non-RNN models cannot directly handle variable-length time series.
        - Time series data is regularly sampled to create fixed-length inputs.
        - For non-RNN "Simple" method, a masking vector is concatenated with the input.
        - PhysioNet dataset:
            - Sampled hourly.
            - Forward or backward propagation used to fill gaps.
        - SVMs use Gaussian RBF kernel (chosen for better performance).
        - scikit-learn is used for implementing and tuning non-RNN models via cross-validation.
        - GRU-Mean has: 100 hidden units for MIMIC-III, 64 hidden units for PhysioNet.
        - Batch normalization and dropout (rate 0.5) are applied to the regressor layer.
        - RNN models are trained using Adam optimizer with early stopping.
        - Keras and Theano are used to implement RNNs.
        - Among all these imputation methods, with LR and SVM, the SoftImpute performed the best.
        - CubicSpline, which captures the temporal structure of the data performed the best with RF, but failed with SVM and GRU. 
        - MissForest provides slightly performed better with GRU models than other additional imputation baselines.


### 2025-05-23
- [x] Attended a meeting (11 AM)
- Meeting discussions:
    - Discussed about the [`STraTS library`](https://github.com/sindhura97/STraTS/tree/main/src)
    - TO DOs:
        - Take a list of admissions and preprocess labevents table data
        - Use chart time
        - Admission file has time of admission time in hr/min/day in admissions table (beginning of admission) -> discretize the timestamp based on hour/day etc. if you have many data in same date take the average
        - Select all features (if you wish to make feature selection)
        - Start with Random Forest
        - Train with oversampling (+) weighted loss function
        - Train it on X, M and Delta with different combinations
        - Then start with on GRU and GRU-D
        - Train it on X, M and Delta with different combinations
        - Compare their perfo- Aggregate by 6 hours in the preprocessing instead of daily -> every two hours/6 hours
- Stratify the data such that the model doesn't see the data that was used during the training
    - To generate Cross Validation Split
- Train model on demographic data age, gender, ethinicity, instead of `target` colummn in the Cohort file.
- Modify pipeline such that:
    - To give different target
    - To change the aggregation window 2, 6 ,12 ,24 hrs
    - To add `delta` table (time since last observed)
- Look at feature importance in Random Forest
    - best model.doc feature importancermances
        - Next part: Patterns of M on different demographic data
- [x] Project Initial Setup
    - [x] Folder and files structuring

---

### 2025-05-29: Project Setup
- [x] Fixed the `.env` variables to be read successfully.
- [x] Solved the issue with heavy data loading by just taking the chunk of the file and saving to a `.parquet` file 
- [x] Solved issues with docker memory size and reconfigured the `.yml` file
- [x] Prepared `14 days prior` and `7 days prior` data for further processing

### 2025-06-01
- [x] Created a pre-processing pipeline for tabular data
- [x] Structured folders to fit a ML pipeline

### 2025-06-02
- [x] Refactored proper data preprocessing  (`dataset.py`) and data loading (`data_loader.py`)
- [x] Configured `config.yml` for easier hyperparameter setting
- [x] Configured `schemas.py` to handle data, model, training, evaluation, logging and experimental config
- [x] The `logging_utils.py` will be used to log into a file and also in the console
- [x] Random Forest model setup 

--- 
### 2025-06-04: Refactoring and debugging
- [x] Debugging `typing`, `parameter initialization`, `file path` errors

### 2025-06-05
- Issues faced and fixed
    - [x] Attribute Errors
    - [x] File path issues. Lesson learnt: avoid using `os.path.join`, instead use `Path(__file__).parent` for more surity.
    - [x] Pydantic validation errors. 

### 2025-06-06
[x] Attended a meeting (11 AM)
- Aggregate by 6 hours in the preprocessing instead of daily -> every two hours/6 hours [2,6,12,24]


- Stratify the data such that the model doesn't see the data (from the same patient) that was used during the training
- To generate Cross Validation Split on the patients instead of admissions
- train/test splits on patients -> train/test splits on admissions for those patients


- Train model on demographic data age, gender, ethnicity, instead of `target` column in the Cohort file. [patient table in mimic, small, can read in pandas]


- Modify pipeline such that:
   - To give different target
   - To change the aggregation window 2, 6 ,12 ,24 hrs
   - To add `delta` table (time since last observed)
   - To train added models (STraTs pipeline - GRU, LSTM) 


- Look at feature importance in Random Forest 
   - best_model.feature_importances_ -> importance value for each feature
https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
oversampling (only training set) imblearn
Class_weight ({0:1, 1:2})
