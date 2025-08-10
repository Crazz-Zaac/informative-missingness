from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    recall_score, precision_score, balanced_accuracy_score, f1_score
)

# define params
SEED = 42
GRID = True

param_grid = {
    'n_estimators': [10, 100, 200],
    'max_depth': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'class_weight': [
        None,
        {0: 0.5, 1: 1},
        {0: 0.5, 1: 1.5},
        {0: 0.5, 1: 2}
    ]
}

best_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'class_weight': {0: 1, 1: 2}
}

# assume df and cohort are loaded correctly with appropriate indices
df = ...  # training data with hadm_id as index
cohort = ...  # cohort dataframe with target column, subject_id and hadm_id

tg = cohort.set_index("hadm_id")["target"].reindex(df.index) #(this reorganise rows and remove any admission without labs)

results, feat_imp = [],[]

# prepare stratified group k-fold splitter
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED  
)

groups = cohort.set_index("hadm_id").reindex(df.index)["subject_id"].values

X = df.values
y = tg.values

for i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):

    # get train/test splits
    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # --- Imputation (no imputation needed in random forest but needed in other models) ---
    # imputer = KNNImputer(n_neighbors=10)
    # x_train = imputer.fit_transform(x_train)
    # x_test = imputer.transform(x_test)

    # --- Scaling (no scaling needed in random forest but needed in other models) ---
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    
    # --- Oversample minority class ---
    oversample = RandomOverSampler(sampling_strategy='minority', random_state=SEED)
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    
    # --- Random Forest training ---
    if GRID:
        rf = GridSearchCV(
            RandomForestClassifier(random_state=SEED),
            param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1
        )
        rf.fit(x_train, y_train)
        best_rf = rf.best_estimator_
        print(f"Best parameters for fold {i+1}:", rf.best_params_)
    else:
        best_rf = RandomForestClassifier(**best_params, random_state=SEED)
        best_rf.fit(x_train, y_train)

    # --- Evaluation ---
    pred = best_rf.predict(x_test)
    prob = best_rf.predict_proba(x_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, prob)
    pr_auc = auc(recall, precision)

    metrics = {
        'fold': i + 1,
        'roc_auc': roc_auc_score(y_test, prob),
        'pr_auc': pr_auc,
        'recall': recall_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred)
    }
    results.append(metrics)
    feat_imp.append(best_rf.feature_importances_)
    print(metrics)


