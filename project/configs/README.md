# Configuration Files (configs/)

- This directory contains configuration files that define all the parameters required for data preprocessing, model setup, training, and evaluation.

- ⚠️ **Important:** You must review and adjust these parameters before starting model training. The settings here directly control how the data is prepared and how models are trained.

---

# Key Sections in the YAML File

- `experiment_name` : A unique identifier for the experiment. Useful for tracking results.

- `data` 
    - `tabular`:
        - `data_path`: Paths to the preprocessed datasets (tabular and temporal).
        - `training_data`: Must be set to match the cohort you want to train the model on (e.g., nf or apl).
        - `window_size`: Should be adjusted to the same window size used during raw data preparation (commonly 7, 14, or 21 days prior). Mismatched values will lead to inconsistent results.
        - `feature_combinations`: Defines which aspects of the data are included:
            - `x`: observed features
            - `m`: missingness indicators
            - `delta`: time elapsed since the last observation
            - You can also combine them (x_m, x_delta, m_delta, x_m_delta). The choice here has a major impact on how the model interprets missing data.
        - `training_feature`: Defines the prediction target (e.g., target, gender, race, anchor_age).
            - age_threshold, insurance_type: Filter cohort subsets (e.g., private vs. non-private).
    - `temporal`:
        - `data_path`: Path to preprocessed temporal data.
        - ⚠️ Currently not in use — this is reserved for temporal ML models, which are not yet implemented in the pipeline.
- `model`
    - Choose the model_type (e.g., RandomForest, GradientBoosting, LogisticRegression, XGBoost, CatBoost).
    - Each model has its own set of fixed_params (always applied) and grid_search_params (used for hyperparameter tuning).
- `training`
    - Defines training behavior, e.g., validation_split.
    - **Note:** This hasn't yet been implement but can be added later
    
- `evaluation`
    - Metrics that will be computed during evaluation (e.g., accuracy, F1-score, precision, recall).

- `logging`
    - Manages logging behavior, including level and format.