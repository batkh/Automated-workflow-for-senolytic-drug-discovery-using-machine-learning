import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress precision zero division warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from utils import (
    load_dataset,
    select_features,
    extract_max_depth,
    tune_and_find_best_model,
    instantiate_model,
    evaluate_predictions
)


def run_experiment(exp_id: int, num_trials: int = 5):
    """Execute a specific experiment routine (1-10)."""
    print(f"\n--- Running Experiment {exp_id:02d} ---")

    # Define Experiment Configuration Map
    # Define Experiment Configuration Map (12 Experiments)
    config_map = {
        # RDKit Experiments
        1:  {"dataset": "RDKit",   "fs": "rf_importance", "optuna": False, "threshold": 0.0},
        2:  {"dataset": "RDKit",   "fs": "mutual_info",   "optuna": False, "threshold": 0.0},
        3:  {"dataset": "RDKit",   "fs": "fisher_score",  "optuna": False, "threshold_range": (0.1, 0.5)},
        4:  {"dataset": "RDKit",   "fs": "rf_importance", "optuna": True,  "threshold": 0.0},
        5:  {"dataset": "RDKit",   "fs": "mutual_info",   "optuna": True,  "threshold": 0.0},
        6:  {"dataset": "RDKit",   "fs": "fisher_score",  "optuna": True,  "threshold_range": (0.1, 0.5)},
        
        # Mordred Experiments
        7:  {"dataset": "Mordred", "fs": "rf_importance", "optuna": False, "threshold": 0.0},
        8:  {"dataset": "Mordred", "fs": "mutual_info",   "optuna": False, "threshold": 0.0},
        9:  {"dataset": "Mordred", "fs": "fisher_score",  "optuna": False, "threshold_range": (0.1, 0.7)},
        10: {"dataset": "Mordred", "fs": "rf_importance", "optuna": True,  "threshold": 0.0},
        11: {"dataset": "Mordred", "fs": "mutual_info",   "optuna": True,  "threshold": 0.0},
        12: {"dataset": "Mordred", "fs": "fisher_score",  "optuna": True,  "threshold_range": (0.1, 0.7)},
    }

    if exp_id not in config_map:
        raise ValueError(f"Invalid experiment ID: {exp_id}. Choose between 1 and 12.")

    cfg = config_map[exp_id]
    
    # Load corresponding data
    train_file = f"{cfg['dataset']}_datasets/training_set_{cfg['dataset']}.csv"
    test_file = f"{cfg['dataset']}_datasets/Test_set_{cfg['dataset']}.csv"
    
    X_var, y, test_df = load_dataset(train_file, test_file)

    train_results, test_results = [], []

    for trial in range(num_trials):
        # Determine feature selection threshold
        if "threshold_range" in cfg:
            thresholds = [
                cfg["threshold_range"][0] + i * (cfg["threshold_range"][1] - cfg["threshold_range"][0]) / num_trials
                for i in range(num_trials)
            ]
            thresh = thresholds[trial]
        else:
            thresh = cfg["threshold"]

        # Feature Selection
        important_features = select_features(X_var, y, method=cfg["fs"], threshold=thresh)
        X_selected = X_var[important_features]

        if not cfg["optuna"]:
            # Standard XGBoost execution without Optuna
            max_depth = extract_max_depth(X_selected, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.3, stratify=y, random_state=42 + trial
            )

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.5,
                max_depth=max_depth,
                n_estimators=100,
                colsample_bytree=0.5,
                random_state=42
            )
            pipeline = make_pipeline(MinMaxScaler(), model)
            pipeline.fit(X_train, y_train)

            y_pred_tr = pipeline.predict(X_test)
            y_proba_tr = pipeline.predict_proba(X_test)[:, 1]
            metrics_tr = evaluate_predictions(y_test, y_pred_tr, y_proba_tr)
            metrics_tr.update({
                'Model': 'XGB',
                'Number_of_selected_features': len(important_features),
                'max_depth': max_depth
            })

            # Test Set Evaluation
            y_test_ext = test_df['Senolytic']
            X_test_ext = test_df[important_features]
            y_pred_te = pipeline.predict(X_test_ext)
            y_proba_te = pipeline.predict_proba(X_test_ext)[:, 1]
            metrics_te = evaluate_predictions(y_test_ext, y_pred_te, y_proba_te)
            metrics_te.update({
                'Model': 'XGB',
                'Number_of_selected_features': len(important_features),
                'max_depth': max_depth
            })

        else:
            # Optuna hyperparameter optimization execution
            best_model_name, best_params = tune_and_find_best_model(X_selected, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.3, stratify=y, random_state=42 + trial
            )

            clf = instantiate_model(best_model_name, best_params)
            pipeline = make_pipeline(MinMaxScaler(), clf)
            pipeline.fit(X_train, y_train)

            y_pred_tr = pipeline.predict(X_test)
            y_proba_tr = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            metrics_tr = evaluate_predictions(y_test, y_pred_tr, y_proba_tr)
            metrics_tr.update({
                'Model': best_model_name,
                'Number_of_selected_features': len(important_features),
                'Best_Parameters': str(best_params)
            })

            # Test Set Evaluation
            y_test_ext = test_df['Senolytic']
            X_test_ext = test_df[important_features]
            y_pred_te = pipeline.predict(X_test_ext)
            y_proba_te = pipeline.predict_proba(X_test_ext)[:, 1] if hasattr(pipeline, "predict_proba") else None
            metrics_te = evaluate_predictions(y_test_ext, y_pred_te, y_proba_te)
            metrics_te.update({
                'Model': best_model_name,
                'Number_of_selected_features': len(important_features),
                'Best_Parameters': str(best_params)
            })

        train_results.append(metrics_tr)
        test_results.append(metrics_te)

    # Convert metric lists to DataFrames
    df_train_res = pd.DataFrame(train_results)
    df_test_res = pd.DataFrame(test_results)

    def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Compute Mean ± Std for numeric metrics and preserve non-numeric metadata."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std()
        
        summary_dict = {}
        # Format numeric metrics as 'Mean ± Std'
        for col in numeric_cols:
            summary_dict[col] = [f"{means[col]:.4f} ± {stds[col]:.4f}"]
        
        # Preserve static metadata from the first trial
        for col in df.columns:
            if col not in numeric_cols:
                summary_dict[col] = [df[col].iloc[0]]
                
        return pd.DataFrame(summary_dict)

    # Generate aggregated dataframes
    df_train_summary = calculate_summary_stats(df_train_res)
    df_test_summary = calculate_summary_stats(df_test_res)

    # Export aggregated results to CSV
    df_train_summary.to_csv(f"output/Method_{exp_id:02d}-Validation_Table1.csv", index=False)
    df_test_summary.to_csv(f"output/Method_{exp_id:02d}-Test_Table2.csv", index=False)

    # Optional: Save raw trial-by-trial logs separately
    df_train_res.to_csv(f"output/Method_{exp_id:02d}-Validation_Table1_raw.csv", index=False)
    df_test_res.to_csv(f"output/Method_{exp_id:02d}-Test_Table2_raw.csv", index=False)

    print(f"Aggregated summary saved to Method_{exp_id:02d}-Table1.csv and Method_{exp_id:02d}-Table2.csv")


if __name__ == "__main__":
    # Run experiment 1 by default, or pass an argument via CLI
    experiment_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_experiment(exp_id=experiment_number, num_trials=5)