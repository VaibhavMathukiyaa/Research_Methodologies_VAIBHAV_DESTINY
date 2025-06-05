import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

def balance_classes(X, y):
    """Balance classes using SMOTE."""
    print("\nClass distribution before balancing:")
    print(pd.Series(y).value_counts())
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("Class distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    
    return X_resampled, y_resampled

def split_data(X, y, test_size=0.2):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Train multiple models with hyperparameter tuning."""
    # Define models with hyperparameter grids
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    # Train and tune each model
    best_models = {}
    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_models

def save_models(models, feature_names, save_dir):
    """Save trained models to disk."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        model_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")
    
    # Save feature names
    feature_path = os.path.join(save_dir, "feature_names.pkl")
    joblib.dump(feature_names, feature_path)
    print(f"Feature names saved: {feature_path}")

def train_and_save_models(X, y, feature_names, models_dir):
    """Complete model training pipeline."""
    # Balance classes
    X_resampled, y_resampled = balance_classes(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    
    # Train models
    best_models = train_models(X_train, y_train)
    
    # Save models
    save_models(best_models, feature_names, models_dir)
    
    return X_train, X_test, y_train, y_test, best_models

if __name__ == "__main__":
    # For testing
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    
    data_dir = os.path.join(os.getcwd(), 'data')
    results_dir = os.path.join(os.getcwd(), 'results')
    models_dir = os.path.join(results_dir, 'models')
    file_path = os.path.join(data_dir, 'diabetes.csv')
    
    # Preprocess data
    data_imputed, imputer = preprocess_data(file_path, results_dir)
    
    # Engineer features
    X_scaled, y, feature_names, scaler = engineer_features(data_imputed, results_dir)
    
    # Save preprocessors
    joblib.dump(imputer, os.path.join(models_dir, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    # Train and save models
    X_train, X_test, y_train, y_test, best_models = train_and_save_models(
        X_scaled, y, feature_names, models_dir
    )
    
    # Save train-test split for evaluation
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    joblib.dump(split_data, os.path.join(models_dir, 'train_test_split.pkl'))
    
    print("\nModel training completed successfully.")