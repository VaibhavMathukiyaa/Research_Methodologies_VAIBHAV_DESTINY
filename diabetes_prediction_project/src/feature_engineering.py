import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import joblib

def create_features(data):
    """Create new features from existing ones."""
    data_featured = data.copy()
    
    # Create ratio features
    data_featured['Glucose_to_BMI_Ratio'] = data_featured['Glucose'] / data_featured['BMI']
    
    # Create interaction features
    data_featured['Age_BMI_Interaction'] = data_featured['Age'] * data_featured['BMI'] / 100
    
    # Log transform skewed features
    data_featured['Insulin_Log'] = np.log1p(data_featured['Insulin'])
    data_featured['DiabetesPedigreeFunction_Log'] = np.log1p(data_featured['DiabetesPedigreeFunction'])
    
    # BMI categories according to WHO
    bins = [0, 18.5, 25, 30, 35, 100]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II_III']
    data_featured['BMI_Category'] = pd.cut(data_featured['BMI'], bins=bins, labels=labels)
    
    # Age groups
    age_bins = [0, 30, 45, 60, 100]
    age_labels = ['Young', 'Middle_Aged', 'Senior', 'Elderly']
    data_featured['Age_Group'] = pd.cut(data_featured['Age'], bins=age_bins, labels=age_labels)
    
    # Convert categorical variables to dummy variables
    data_featured = pd.get_dummies(data_featured, columns=['BMI_Category', 'Age_Group'], drop_first=False)
    
    print(f"Data shape after feature engineering: {data_featured.shape}")
    print(f"New features created: {set(data_featured.columns) - set(data.columns)}")
    
    return data_featured

def select_features(data, save_dir):
    """Select relevant features and remove highly correlated ones."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Separate features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Check for high correlation between features
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix After Feature Engineering')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrix_features.png'))
    
    # Remove highly correlated features (threshold > 0.8)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    
    print(f"Highly correlated features to be dropped: {to_drop}")
    X_reduced = X.drop(columns=to_drop)
    
    # Visualize feature correlations with target
    plt.figure(figsize=(12, 8))
    correlation_with_target = pd.DataFrame(
        {'correlation': data.corr()['Outcome'].drop('Outcome')}
    ).sort_values('correlation', ascending=False)
    
    sns.barplot(x=correlation_with_target.index, y='correlation', data=correlation_with_target)
    plt.xticks(rotation=90)
    plt.title('Feature Correlations with Diabetes Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'target_correlations.png'))
    
    return X_reduced, y

def scale_features(X):
    """Standardize features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled_df, scaler

def engineer_features(data, results_dir=None):
    """
    Apply advanced feature engineering techniques.
    """
    # Separate features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Create basic features
    X_engineered = X.copy()
    
    # 1. Ratio features
    X_engineered['Glucose_to_BMI_Ratio'] = X['Glucose'] / X['BMI']
    X_engineered['Insulin_to_Glucose_Ratio'] = X['Insulin'] / (X['Glucose'] + 1)  # Add 1 to avoid division by zero
    X_engineered['BMI_to_Age_Ratio'] = X['BMI'] / X['Age']
    X_engineered['Glucose_to_Insulin_Ratio'] = X['Glucose'] / (X['Insulin'] + 1)
    
    # 2. Interaction features
    X_engineered['Age_BMI_Interaction'] = X['Age'] * X['BMI'] / 100
    X_engineered['Glucose_BMI_Interaction'] = X['Glucose'] * X['BMI'] / 100
    X_engineered['Glucose_BMI_Age_Interaction'] = X['Glucose'] * X['BMI'] * X['Age'] / 10000
    
    # 3. Logarithmic transformations (adding 1 to handle zeros)
    X_engineered['Insulin_Log'] = np.log1p(X['Insulin'])
    X_engineered['DiabetesPedigreeFunction_Log'] = np.log1p(X['DiabetesPedigreeFunction'])
    X_engineered['BMI_Log'] = np.log1p(X['BMI'])
    
    # 4. Polynomial features for key variables
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    poly_features = pd.DataFrame(
        poly.fit_transform(X[['Glucose', 'BMI', 'Age']]),
        columns=poly.get_feature_names_out(['Glucose', 'BMI', 'Age'])
    )
    # Keep only interaction terms to avoid duplicates
    interaction_columns = [col for col in poly_features.columns if ' ' in col]
    for col in interaction_columns:
        new_col_name = col.replace(' ', '_')
        X_engineered[f'Poly_{new_col_name}'] = poly_features[col]
    
    # 5. BMI categories (categorical feature)
    bins = [0, 18.5, 25, 30, 35, 100]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II_III']
    X_engineered['BMI_Category'] = pd.cut(X['BMI'], bins=bins, labels=labels)
    
    # One-hot encode BMI categories
    bmi_dummies = pd.get_dummies(X_engineered['BMI_Category'], prefix='BMI_Category')
    X_engineered = pd.concat([X_engineered, bmi_dummies], axis=1)
    X_engineered.drop('BMI_Category', axis=1, inplace=True)
    
    # 6. Age groups (categorical feature)
    age_bins = [0, 30, 45, 60, 100]
    age_labels = ['Young', 'Middle_Aged', 'Senior', 'Elderly']
    X_engineered['Age_Group'] = pd.cut(X['Age'], bins=age_bins, labels=age_labels)
    
    # One-hot encode Age groups
    age_dummies = pd.get_dummies(X_engineered['Age_Group'], prefix='Age_Group')
    X_engineered = pd.concat([X_engineered, age_dummies], axis=1)
    X_engineered.drop('Age_Group', axis=1, inplace=True)
    
    # 7. Advanced features based on medical knowledge
    # Homeostasis model assessment (HOMA-IR) - approximation
    X_engineered['HOMA_IR'] = (X['Glucose'] * X['Insulin']) / 405
    
    # Quantitative Insulin Sensitivity Check Index (QUICKI) - approximation
    X_engineered['QUICKI'] = 1 / (np.log10(X['Insulin'] + 1) + np.log10(X['Glucose']))
    
    # Print feature engineering results
    print(f"Data shape after feature engineering: {X_engineered.shape}")
    print(f"New features created: {set(X_engineered.columns) - set(X.columns)}")
    
    # 8. Feature selection to remove highly correlated features
    corr_matrix = X_engineered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    print(f"Highly correlated features to be dropped: {to_drop}")
    X_engineered = X_engineered.drop(to_drop, axis=1)
    
    # 9. Feature selection using mutual information
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    # Select top 20 features based on mutual information
    selector = SelectKBest(mutual_info_classif, k=20)
    X_selected = selector.fit_transform(X_engineered, y)
    
    # Get selected feature names
    selected_features = X_engineered.columns[selector.get_support()]
    print(f"Top features selected by mutual information: {list(selected_features)}")
    
    X_engineered = X_engineered[selected_features]
    
    # 10. Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_engineered)
    
    # Convert back to DataFrame with feature names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_engineered.columns)
    
    # Save feature names and scaler if results_dir is provided
    if results_dir:
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        feature_names = list(X_engineered.columns)
        joblib.dump(feature_names, os.path.join(results_dir, 'models/feature_names.pkl'))
        joblib.dump(scaler, os.path.join(results_dir, 'models/scaler.pkl'))
    
    return X_scaled_df, y, list(X_engineered.columns), scaler


if __name__ == "__main__":
    # For testing
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_preprocessing import preprocess_data
    
    data_dir = os.path.join(os.getcwd(), 'data')
    results_dir = os.path.join(os.getcwd(), 'results')
    file_path = os.path.join(data_dir, 'diabetes.csv')
    
    # Preprocess data
    data_imputed, _ = preprocess_data(file_path, results_dir)
    
    # Engineer features
    X_scaled, y, feature_names, scaler = engineer_features(data_imputed, results_dir)
    
    print("\nFeature engineering completed successfully.")
    print(f"Final feature set: {feature_names}")