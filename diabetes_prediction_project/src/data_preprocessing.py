import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import os
import joblib


def load_data(file_path):
    """Load the diabetes dataset."""
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {data.shape}")
    return data

def explore_data(data, save_dir):
    """Perform basic exploratory data analysis."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(data.isnull().sum())
    
    # Check for zeros in columns where zero is not physiologically valid
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("\nZero values in columns where zero is not physiologically valid:")
    for column in zero_columns:
        zero_count = (data[column] == 0).sum()
        print(f"{column}: {zero_count} zeros ({zero_count/len(data)*100:.2f}%)")
    
    # Target variable distribution
    print("\nTarget variable distribution:")
    print(data['Outcome'].value_counts())
    print(f"Percentage of diabetic cases: {data['Outcome'].mean() * 100:.2f}%")
    
    # Visualize feature distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(data.columns[:-1], 1):
        plt.subplot(3, 3, i)
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'))
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
    
    # Box plots by outcome
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(data.columns[:-1], 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x='Outcome', y=column, data=data)
        plt.title(f'{column} by Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxplots_by_outcome.png'))
    
    return

def handle_missing_values(data):
    """Replace zeros with NaN for columns where zero is not a valid value."""
    data_processed = data.copy()
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in zero_columns:
        data_processed[column] = data_processed[column].replace(0, np.nan)
    
    print("\nMissing values after replacing zeros with NaN:")
    print(data_processed.isnull().sum())
    
    return data_processed

def impute_missing_values(data, save_dir):
    """Impute missing values using KNN imputation."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Store original data for columns with missing values
    columns_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    original_data = {col: data[col].dropna() for col in columns_with_missing}
    
    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = data.copy()
    data_imputed.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])
    
    # Visualize imputation results
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns_with_missing, 1):
        plt.subplot(3, 2, i)
        sns.histplot(original_data[column], kde=True, color='blue', alpha=0.5, label='Original (non-zero)')
        sns.histplot(data_imputed[column], kde=True, color='red', alpha=0.5, label='After Imputation')
        plt.title(f'Distribution of {column} - Before vs After Imputation')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'imputation_comparison.png'))
    
    return data_imputed, imputer

def preprocess_data(data_path, results_dir=None):
    """
    Preprocess the diabetes dataset with advanced techniques.
    """
    # Load data
    data = pd.read_csv(data_path)
    print(f"Dataset loaded with shape: {data.shape}")
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(data.isnull().sum())
    
    # Identify zero values in columns where zero is not physiologically valid
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\nZero values in columns where zero is not physiologically valid:")
    for column in zero_columns:
        zero_count = (data[column] == 0).sum()
        zero_percent = zero_count / len(data) * 100
        print(f"{column}: {zero_count} zeros ({zero_percent:.2f}%)")
    
    # Display target variable distribution
    print("\nTarget variable distribution:")
    print(data['Outcome'].value_counts())
    print(f"Percentage of diabetic cases: {data['Outcome'].mean()*100:.2f}%")
    
    # Replace zeros with NaN for columns where zero is not physiologically valid
    for column in zero_columns:
        data[column] = data[column].replace(0, np.nan)
    
    print("\nMissing values after replacing zeros with NaN:")
    print(data.isnull().sum())
    
    # Advanced imputation using Iterative Imputer (MICE)
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor
    
    # Modified imputer parameters to address convergence warning
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),  # Simpler model
        max_iter=25,  # Increased from 10 to 25
        tol=0.01,     # More lenient tolerance (default is 0.001)
        random_state=42,
        skip_complete=True
    )
    
    # Fit and transform the data
    # First, separate features and target to avoid any potential issues
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Impute only the features (not the target)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    
    # Recombine with the target
    data_imputed = pd.concat([X_imputed, y], axis=1)
    
    # Save preprocessed data and imputer if results_dir is provided
    if results_dir:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(results_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save data and imputer
        data_imputed.to_csv(os.path.join(results_dir, 'preprocessed_data.csv'), index=False)
        joblib.dump(imputer, os.path.join(models_dir, 'imputer.pkl'))
    
    return data_imputed, imputer

if __name__ == "__main__":
    # Test the preprocessing pipeline
    data_dir = os.path.join(os.getcwd(), 'data')
    results_dir = os.path.join(os.getcwd(), 'results')
    file_path = os.path.join(data_dir, 'diabetes.csv')
    
    data_imputed, imputer = preprocess_data(file_path, results_dir)
    print("\nPreprocessing completed successfully.")
    print(f"Processed data shape: {data_imputed.shape}")