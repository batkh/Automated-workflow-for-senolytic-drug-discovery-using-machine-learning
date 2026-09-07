import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress precision zero division warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score
)
from skfeature.function.similarity_based import fisher_score
import optuna

# Suppress Optuna verbosity logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def load_dataset(train_path: str, test_path: str):
    """Load, clean infinite/extreme values, and preprocess datasets safely for float32."""
    
    def sanitize_dataframe(df):
        # 1. Drop non-feature metadata
        if 'SMILES' in df.columns:
            df = df.drop(['SMILES'], axis=1)
        if 'Name' in df.columns:
            df = df.set_index('Name')

        # 2. Coerce string error messages to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # 3. Replace positive/negative infinity with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # 4. Clip extreme values to prevent float32 overflow
        f32_max = np.finfo(np.float32).max
        f32_min = np.finfo(np.float32).min
        df = df.clip(lower=f32_min, upper=f32_max)

        return df

    # Load and clean training data
    df_train = pd.read_csv(train_path, sep=",")
    df_train = shuffle(df_train, random_state=42)
    df_train = sanitize_dataframe(df_train)
    
    # Drop columns containing NaN or missing values derived from cleaning
    df_train = df_train.dropna(axis=1)

    X_train = df_train.drop(['Senolytic'], axis=1).astype(np.float32)
    y_train = df_train["Senolytic"].astype(int)

    # Load and clean testing data
    df_test = pd.read_csv(test_path, sep=",")
    df_test = shuffle(df_test, random_state=42)
    df_test = sanitize_dataframe(df_test)
    
    # Fill remaining NaNs on test features with 0 to maintain shape alignment
    df_test = df_test.fillna(0)

    # Align columns in df_test with X_train (ensuring identical feature alignment)
    test_target = df_test['Senolytic'].astype(int) if 'Senolytic' in df_test.columns else None
    df_test = df_test.reindex(columns=X_train.columns, fill_value=0).astype(np.float32)
    if test_target is not None:
        df_test['Senolytic'] = test_target

    return X_train, y_train, df_test

def min_max_normalization(scores: np.ndarray) -> np.ndarray:
    """Normalize input array using Min-Max scaling."""
    min_score, max_score = np.min(scores), np.max(scores)
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def select_features(X: pd.DataFrame, y: pd.Series, method: str, threshold: float = 0.0) -> list:
    """Extract key features using specified selection strategy."""
    if method == "rf_importance":
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        important_features = importances[importances > threshold].index.tolist()

    elif method == "mutual_info":
        importance = mutual_info_classif(X, y, random_state=42)
        feat_important = pd.Series(importance, index=X.columns)
        important_features = feat_important[feat_important > threshold].index.tolist()

    elif method == "fisher_score":
        ranks = fisher_score.fisher_score(X.to_numpy(), y.to_numpy())
        normalized_ranks = min_max_normalization(ranks)
        feat_importances = pd.Series(normalized_ranks, index=X.columns)
        important_features = feat_importances[feat_importances > threshold].index.tolist()

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    return important_features


def extract_max_depth(X: pd.DataFrame, y: pd.Series) -> int:
    """Find optimal XGBoost tree depth based on cross-validated precision scores."""
    tree_depths = list(range(1, 11))
    precision_means = []

    for depth in tree_depths:
        current_model = xgb.XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.5,
            max_depth=depth,
            n_estimators=100,
            colsample_bytree=0.5,
            random_state=42
        )
        pipeline = make_pipeline(MinMaxScaler(), current_model)
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='precision')
        precision_means.append(np.mean(scores))

    return tree_depths[np.argmax(precision_means)]


def optuna_objective(trial, classifier_name: str, x_train, y_train, x_val, y_val):
    """Objective function for Optuna hyperparameter optimization."""
    if classifier_name == 'SVM':
        svr_c = trial.suggest_float('C', 1, 100)
        svr_gamma = trial.suggest_float('gamma', 1e-4, 10, log=True)
        svr_kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        clf = SVC(C=svr_c, kernel=svr_kernel, gamma=svr_gamma, random_state=42)

    elif classifier_name == 'RandomForest':
        n_estimators = trial.suggest_int("n_estimators", 50, 300, log=True)
        max_depth = trial.suggest_int("max_depth", 10, 100)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    elif classifier_name == "NaiveBayes":
        var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1)
        clf = GaussianNB(var_smoothing=var_smoothing)

    elif classifier_name == "XGB":
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.9)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

    elif classifier_name == 'LR':
        max_iter = trial.suggest_int('max_iter', 100, 300)
        c_values = trial.suggest_float('C', 0.1, 100, log=True)
        solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear'])
        clf = LogisticRegression(C=c_values, max_iter=max_iter, solver=solver, random_state=42)

    pipeline = make_pipeline(MinMaxScaler(), clf)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_val)

    return precision_score(y_val, y_pred, zero_division=0)


def tune_and_find_best_model(X: pd.DataFrame, y: pd.Series):
    """Run Optuna tuning with Nested Cross-Validation across multiple classifiers."""
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    classifiers = ['SVM', 'NaiveBayes', 'RandomForest', 'XGB', 'LR']
    
    best_overall_model = None
    best_overall_params = None
    best_overall_score = -1

    for classifier_name in classifiers:
        precision_scores = []
        last_params = {}

        for train_idx, val_idx in outer_cv.split(X, y):
            x_train_out, x_val_out = X.iloc[train_idx], X.iloc[val_idx]
            y_train_out, y_val_out = y.iloc[train_idx], y.iloc[val_idx]

            study = optuna.create_study(direction='maximize')
            obj = lambda trial: optuna_objective(
                trial, classifier_name, x_train_out, y_train_out, x_val_out, y_val_out
            )
            study.optimize(obj, n_trials=10, n_jobs=-1)
            
            precision_scores.append(study.best_value)
            last_params = study.best_params

        mean_score = np.mean(precision_scores)
        if mean_score > best_overall_score:
            best_overall_score = mean_score
            best_overall_model = classifier_name
            best_overall_params = last_params

    return best_overall_model, best_overall_params


def instantiate_model(model_name: str, params: dict):
    """Instantiate a model instance from configuration dictionary parameters."""
    if model_name == 'XGB':
        return xgb.XGBClassifier(**params, random_state=42)
    elif model_name == 'SVM':
        return SVC(**params, probability=True, random_state=42)
    elif model_name == 'LR':
        return LogisticRegression(**params, random_state=42)
    elif model_name == 'NaiveBayes':
        return GaussianNB(**params)
    elif model_name == 'RandomForest':
        return RandomForestClassifier(**params, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def evaluate_predictions(y_true, y_pred, y_proba=None):
    """Calculate model performance metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan

    metrics = {
        'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1_score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'ROC_AUC': roc_auc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    return metrics