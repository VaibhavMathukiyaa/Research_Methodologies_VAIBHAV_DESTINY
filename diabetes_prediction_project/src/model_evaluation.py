import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, log_loss  # Add these metrics
)

def evaluate_models(models, X_test, y_test):
    """Evaluate multiple models on test data with advanced metrics."""
    results = {}
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        # Calculate additional metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        neg_log_loss = log_loss(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'MCC': mcc,
            'Log Loss': neg_log_loss
        }
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    return results_df

def plot_confusion_matrices(models, X_test, y_test, save_dir):
    """Plot confusion matrices for all models."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{name.replace(" ", "_").lower()}.png'))
        plt.close()

def plot_roc_curves(models, X_test, y_test, save_dir):
    """Plot ROC curves for all models."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()

def plot_precision_recall_curves(models, X_test, y_test, save_dir):
    """Plot precision-recall curves for all models."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'))
    plt.close()

def analyze_feature_importance(models, X_train, X_test, feature_names, save_dir):
    """Analyze and visualize feature importance."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance - {name}')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'feature_importance_{name.replace(" ", "_").lower()}.png'))
            plt.close()
            
            # Save feature importance to CSV
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            importance_df.to_csv(os.path.join(save_dir, f'feature_importance_{name.replace(" ", "_").lower()}.csv'), index=False)
        
        elif name == 'Logistic Regression':
            # For logistic regression
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Coefficients - {name}')
            plt.bar(range(X_train.shape[1]), coef[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'feature_coefficients_{name.replace(" ", "_").lower()}.png'))
            plt.close()
            
            # Save coefficients to CSV
            coef_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Coefficient': coef[indices]
            })
            coef_df.to_csv(os.path.join(save_dir, f'feature_coefficients_{name.replace(" ", "_").lower()}.csv'), index=False)

def shap_analysis(models, X_train, X_test, y_test, feature_names, save_dir):
    """Perform SHAP analysis for model interpretability."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Select best model for SHAP analysis
    best_model_name = None
    best_auc = 0
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
    
    print(f"\nPerforming SHAP analysis for best model: {best_model_name}")
    best_model = models[best_model_name]
    
    # Create SHAP explainer
    try:
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
            
            # If shap_values is a list (for binary classification), take the one for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # For non-tree models, use KernelExplainer with a subset of training data
            X_train_summary = shap.kmeans(X_train, 100)
            explainer = shap.KernelExplainer(best_model.predict_proba, X_train_summary)
            shap_values = explainer.shap_values(X_test)
            
            # Take the values for positive class
            shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_summary.png'))
        plt.close() 
        
        # Calculate feature importance from SHAP values
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': feature_importance
        })
        importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
        importance_df.to_csv(os.path.join(save_dir, 'shap_feature_importance.csv'), index=False)
        
        # Create a simple bar plot of SHAP importance
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'][:10], importance_df['SHAP_Importance'][:10])
        plt.xlabel('mean(|SHAP value|)')
        plt.title('Top 10 Features by SHAP Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_importance_bar.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        print("Skipping detailed SHAP analysis due to errors.")
    
    return best_model_name, best_model

def evaluate_and_analyze_models(models, X_train, X_test, y_train, y_test, feature_names, results_dir):
    """Complete model evaluation pipeline."""
    # Create results directories
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Evaluate models
    results_df = evaluate_models(models, X_test, y_test)
    print("\nModel Evaluation Results:")
    print(results_df)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(results_dir, 'model_evaluation_results.csv'))
    
    # Plot confusion matrices
    plot_confusion_matrices(models, X_test, y_test, figures_dir)
    
    # Plot ROC curves
    plot_roc_curves(models, X_test, y_test, figures_dir)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(models, X_test, y_test, figures_dir)
    
    # Analyze feature importance
    analyze_feature_importance(models, X_train, X_test, feature_names, figures_dir)
    
    # Perform SHAP analysis
    best_model_name, best_model = shap_analysis(models, X_train, X_test, y_test, feature_names, figures_dir)
    
    # Generate classification reports
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        with open(os.path.join(results_dir, f'classification_report_{name.replace(" ", "_").lower()}.txt'), 'w') as f:
            f.write(f"Classification Report - {name}\n\n")
            f.write(report)
    
    return best_model_name, best_model

if __name__ == "__main__":
    # For testing
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    results_dir = os.path.join(os.getcwd(), 'results')
    models_dir = os.path.join(results_dir, 'models')
    
    # Load saved models and data
    split_data = joblib.load(os.path.join(models_dir, 'train_test_split.pkl'))
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    
    # Load models
    models = {}
    for model_name in ['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost']:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            models[model_name.replace('_', ' ').title()] = joblib.load(model_path)
    
    # Evaluate and analyze models
    best_model_name, best_model = evaluate_and_analyze_models(
        models, X_train, X_test, y_train, y_test, feature_names, results_dir
    )
    
    print(f"\nBest performing model: {best_model_name}")
    print("\nModel evaluation completed successfully")