import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler
import warnings
from scipy.special import expit
from pathlib import Path
import pickle
import time
import torch
from tqdm import tqdm
from .plot import visualize_results

# ------------------------------- Configuration -------------------------------
# Paths relative to project root (parent of code/), so they work regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILE_NAME = "nba_ml_matchups.csv"
# FILE_NAME = "nba_ml_games.csv"  # Alternative: individual team perspective
DATA_DIR = _PROJECT_ROOT / "data"
OUTPUT_DIR = _PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- Utility Functions -------------------------------

def save_models(models_dict: dict) -> None:
    """Persist trained models to disk under OUTPUT_DIR."""
    for name, model in models_dict.items():
        model_path = OUTPUT_DIR / "models" / f"{name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    print(f"Saved {len(models_dict)} models to {OUTPUT_DIR / 'models'}")
    

def load_models(models_dir: Path | str | None = None, model_names: list[str] | None = None) -> dict:
    """Load trained models from disk.
    
    Args:
        models_dir: Directory containing .pkl files. If None, uses OUTPUT_DIR / "models".
        model_names: If provided, only load these models (by .pkl filename stem). If None, load all.
    """
    directory = Path(models_dir) if models_dir is not None else OUTPUT_DIR / "models"
    models_dict = {}
    names_set = set(model_names) if model_names is not None else None
    for model_path in directory.glob("*.pkl"):
        if names_set is not None and model_path.stem not in names_set:
            continue
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        models_dict[model_path.stem] = model
    print(f"Loaded {len(models_dict)} models from {directory}")
    return models_dict

def load_results() -> list:
    """Load evaluation results from disk under OUTPUT_DIR."""
    results_path = OUTPUT_DIR / "results.csv"
    results = pd.read_csv(results_path)
    return results

def save_results(results, name: str = "results") -> None:
    """Persist evaluation results (list of dicts or DataFrame) as a CSV."""
    if isinstance(results, pd.DataFrame):
        results_df = results
        if results_df.empty:
            print(f"Warning: No results to save for {name}")
            return
    else:
        if not results:
            print(f"Warning: No results to save for {name}")
            return
        results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / f"{name}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

"""
Features are inferred from the dataset columns to avoid hardcoding.

Non-feature columns (metadata / IDs / target):
- GAME_DATE_EST, GAME_ID, SEASON, HOME_TEAM_ID, OPP_TEAM_ID
- TARGET

Everything else is treated as a feature.
"""

# Optional override: populate this list if you want to force a fixed subset.
FEATURES: list[str] = []

# Columns that are not used as features
INFO_COLS = ['GAME_DATE_EST', 'GAME_ID', 'SEASON', 'HOME_TEAM_ID', 'OPP_TEAM_ID']

TARGET = 'WL'


def get_feature_columns(df_model: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    """Infer feature columns from df_model unless an explicit list is provided."""
    if features is None:
        features = FEATURES if FEATURES else None
    if features:
        return list(features)
    exclude = set(INFO_COLS + [TARGET])
    return [c for c in df_model.columns if c not in exclude]

# Model names to train - comment out or remove to exclude models
MODEL_NAMES = [
    'Dummy Baseline',
    'Logistic Regression',
    'Decision Tree',
    'K-Nearest Neighbors',
    'Random Forest',
    'XGBoost'
]

PARAMS = {
    'Dummy Baseline': {
        'strategy': 'most_frequent'
    },
    'Logistic Regression': {
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs'
    },
    'Decision Tree': {
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'K-Nearest Neighbors': {
        'n_neighbors': 50,
        'weights': 'uniform',
        'algorithm': 'auto',
        'metric': 'minkowski',
        'n_jobs': -1
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
}

def init_models(params: dict = None, load: bool = False, load_from_dir: Path | str | None = None, model_names: list[str] | None = None) -> dict:
    """Instantiate models with their configured hyperparameters.
    
    Args:
        params: Model name -> hyperparams dict. If None, uses PARAMS from config.
            The keys of params define which models are created.
        load: If True, load persisted models from disk instead of building from params.
        load_from_dir: If load is True, directory containing .pkl files (e.g. OUTPUT_DIR / "no_fe" / "models").
            If None, uses default OUTPUT_DIR / "models".
        model_names: If provided, only these models are loaded (when load=True) or built (when load=False).
            Must match keys in params when load=False. If None, all available models are used.
    
    Returns:
        Dictionary mapping model names to instantiated model objects.
    """
    if params is None:
        params = PARAMS

    if load:
        models_dict = load_models(models_dir=load_from_dir, model_names=model_names)
        return models_dict
    
    names_iter = model_names if model_names is not None else params
    # Mapping from model names to their constructors
    model_constructors = {
        'Dummy Baseline': DummyClassifier,
        'Logistic Regression': LogisticRegression,
        'Decision Tree': DecisionTreeClassifier,
        'K-Nearest Neighbors': KNeighborsClassifier,
        'Support Vector Machine': SVC,
        'Random Forest': RandomForestClassifier,
        'XGBoost': xgb.XGBClassifier
    }
    
    models_dict = {}
    for name in names_iter:
        if name not in params:
            raise ValueError(f"Model '{name}' not found in params configuration")
        if name not in model_constructors:
            raise ValueError(f"Model '{name}' not found in model constructors")
        
        # Special handling for XGBoost to enable GPU if available
        if name == 'XGBoost':
            xgb_params = params[name].copy()
            if torch.cuda.is_available():
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda'
                print(">>> XGBoost: GPU acceleration enabled")
            else:
                xgb_params['tree_method'] = 'hist'
            models_dict[name] = model_constructors[name](**xgb_params)
        else:
            models_dict[name] = model_constructors[name](**params[name])
    
    return models_dict

# Models that require scaled features (same as in train_models)
_SCALED_MODELS = {'Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine'}


def _get_pred_proba(model, X):
    """Get probability estimates for the positive class.

    Falls back to decision_function + sigmoid for models without
    predict_proba (e.g. SVC with probability=False).
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        return expit(model.decision_function(X))
    else:
        raise ValueError("Model has neither predict_proba nor decision_function")


def tune_params(
    params_grid: dict,
    df_model: pd.DataFrame,
    *,
    search_type: str = 'grid',
    n_iter: int = 20,
    random_state: int = 42,
    metric: str = 'log_loss',
    verbose: int = 1,
) -> dict:
    """
    Tune hyperparameters using the same walk-forward validation as train_models.

    Model names and base config come from params_grid keys and PARAMS; base
    models are created via init_models so behaviour matches training.

    Parameters
    ----------
    params_grid : dict
        Map model_name -> param grid (e.g. {'max_depth': [3,5], 'n_estimators': [100,200]}).
        Model names must exist in PARAMS and in init_models' constructors.
        Entries with empty grid are skipped (return {} for that model).
    df_model : pd.DataFrame
        Dataset with SEASON, FEATURES columns and TARGET; same as used for train_models.
    search_type : str
        'grid' = exhaustive ParameterGrid; 'random' = ParameterSampler with n_iter.
    n_iter : int
        Number of random combinations when search_type='random'.
    random_state : int
        Used for ParameterSampler and for reproducible fits (same as training).
    metric : str
        Metric to minimize: 'log_loss' (default).
    verbose : int
        Verbosity level (e.g. tqdm when > 0).

    Returns
    -------
    dict
        best_params_per_model: { model_name: best_params_dict }.
    """
    features = get_feature_columns(df_model)
    seasons = sorted(df_model['SEASON'].unique())
    seasons_to_process = [s for s in seasons[-8:] if len(df_model[df_model['SEASON'] == s]) > 0]
    if not seasons_to_process:
        return {name: {} for name in params_grid}

    # Metric: lower is better
    if metric == 'log_loss':
        def score_one(y_true, pred_proba):
            if len(np.unique(y_true)) < 2:
                return np.nan
            return log_loss(y_true, pred_proba)
    else:
        raise ValueError(f"metric must be 'log_loss'; got {metric!r}")

    best_by_model = {}

    for name, grid in params_grid.items():
        if not grid:
            best_by_model[name] = {}
            if verbose:
                print(f"[tune_params] No grid for '{name}', skipping.")
            continue

        base_model = init_models(params={name: PARAMS[name]}, load=False)[name]
        if search_type == 'grid':
            param_list = list(ParameterGrid(grid))
        else:
            param_list = list(ParameterSampler(grid, n_iter=n_iter, random_state=random_state))

        if verbose:
            print(f"\n>>> Tuning '{name}' ({len(param_list)} configs, search_type={search_type})")

        # Pre-cache season splits and scaling (done once per model, not per config)
        season_data = {}
        for season in seasons_to_process:
            train = df_model[df_model['SEASON'] < season]
            test = df_model[df_model['SEASON'] == season]
            
            if name == 'XGBoost':
                n_pos = (train['WL'] == 1).sum()
                n_neg = (train['WL'] == 0).sum()
                scale_pos_weight_val = round(n_neg / n_pos, 2)
                params['scale_pos_weight'] = scale_pos_weight_val
            
            X_train = train[features]
            y_train = train[TARGET]
            X_test = test[features]
            y_test = test[TARGET]
            if name in _SCALED_MODELS:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                season_data[season] = (X_train_s, y_train, X_test_s, y_test)
            else:
                season_data[season] = (X_train, y_train, X_test, y_test)

        best_score = np.inf
        best_params = None
        iterator = tqdm(param_list, desc=name, unit="config") if verbose else param_list

        for params in iterator:
            model = clone(base_model)
            model.set_params(**params)
            season_scores = []
            for season in seasons_to_process:
                X_tr, y_tr, X_te, y_te = season_data[season]
                # Suppress sklearn parallel warnings during tuning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", 
                        message=".*sklearn.utils.parallel.delayed.*",
                        category=UserWarning)
                    model.fit(X_tr, y_tr)
                pred_proba = _get_pred_proba(model, X_te)
                s = score_one(y_te, pred_proba)
                season_scores.append(s)
            avg = np.nanmean(season_scores)
            if not np.isnan(avg) and avg < best_score:
                best_score = avg
                best_params = dict(params)

        best_by_model[name] = best_params if best_params is not None else {}
        if verbose and best_params is not None:
            print(f"    best {metric}: {best_score:.5f} -> {best_params}")

    return best_by_model

def train_models(models_dict: dict, df_model: pd.DataFrame, load: bool = False) -> list:
    """Run walk-forward validation across seasons and return a list of metric dicts."""
    features = get_feature_columns(df_model)
    seasons = sorted(df_model['SEASON'].unique())
    results = []

    if load:
        results = load_results()
        return results
    
    print(f"\n>>> Starting Walk-Forward Validation...")
    seasons_to_process = [s for s in seasons[2:] if len(df_model[df_model['SEASON'] == s]) > 0]

    # Streaming results path (append as we go so you can monitor runtime per model)
    results_path = OUTPUT_DIR / "results.csv"
    if results_path.exists():
        # Start a fresh file for this run; old results are assumed to have been archived if needed
        results_path.unlink()

    # ------------------------------- Training Loop -------------------------------
    
    for season in tqdm(seasons_to_process, desc="Seasons", unit="season"):
        # Temporal Split (Walk-Forward)
        train = df_model[df_model['SEASON'] < season]
        test = df_model[df_model['SEASON'] == season]
        
        X_train = train[features]
        y_train = train[TARGET]
        X_test = test[features]
        y_test = test[TARGET]

        # SCALING: Essential for linear / distance-based models (done once per season)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Loop over models (sequential)
        for name, model in tqdm(models_dict.items(), desc=f"  Season {season}", leave=False, unit="model"):
            t0 = time.perf_counter()

            # Use scaled data for models that are sensitive to feature scales
            if name in _SCALED_MODELS:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                pred_proba = _get_pred_proba(model, X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                pred_proba = _get_pred_proba(model, X_test)

            elapsed = time.perf_counter() - t0

            acc = accuracy_score(y_test, preds)
            # Use macro F1 so both classes count equally; avoids Dummy (most_frequent) dominating
            # when the majority class is positive, which would give inflated binary F1.
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)

            # FPR and FNR from confusion matrix
            cm = confusion_matrix(y_test, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            # Guard metrics that require both classes (degenerate seasons)
            if len(np.unique(y_test)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_test, pred_proba)

            row = {
                'Season': season, 
                'Model': name, 
                'Accuracy': acc,
                'F1': f1,
                'AUC': auc,
                'FPR': fpr,
                'FNR': fnr,
                'Time_s': round(elapsed, 3),
            }

            # Keep in-memory list (for notebook use)
            results.append(row)

            # Append immediately to CSV so you can inspect progress in real time
            row_df = pd.DataFrame([row])
            header = not results_path.exists()
            row_df.to_csv(results_path, mode="a", header=header, index=False)
        
    return results


def main():
    df_games = pd.read_csv(DATA_DIR / FILE_NAME)

    # Clean NaNs
    features = get_feature_columns(df_games)
    df_model = df_games.dropna(subset=features).reset_index(drop=True)

    # Data diagnostics
    print(f"\n>>> Data Summary:")
    print(f"   Total rows: {len(df_model):,}")
    print(f"   Features used: {len(FEATURES)}")
    print(f"   Seasons: {df_model['SEASON'].min()} - {df_model['SEASON'].max()}")
    print(f"   Target distribution: {df_model[TARGET].value_counts().to_dict()}")
    print(f"   Target balance: {df_model[TARGET].mean():.3f} (1 = win)")
    print(f"\n>>> Models (from config): {', '.join(PARAMS)}")

    # Initialise, train, evaluate and persist
    models_dict = init_models(load=True)
    results = train_models(models_dict, df_model, load = True)
    # save_models(models_dict)
    # save_results(results, name="results")
    visualize_results(results, OUTPUT_DIR)

if __name__ == "__main__":
    main()