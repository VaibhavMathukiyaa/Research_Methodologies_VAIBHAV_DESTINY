import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def debug_prediction():
    """Debug the model prediction issue."""
    print("\n" + "="*50)
    print("DIABETES PREDICTION DEBUG")
    print("="*50)
    
    # Find the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'src':
        project_dir = os.path.dirname(current_dir)
    else:
        project_dir = current_dir
    
    print(f"Project directory: {project_dir}")
    
    # Paths
    models_dir = os.path.join(project_dir, 'results', 'models')
    print(f"Models directory: {models_dir}")
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        return
    
    # List available model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    print(f"Available model files: {model_files}")
    
    # Load model
    model_path = os.path.join(models_dir, 'random_forest.pkl')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        # Try to find another model
        for model_file in model_files:
            if model_file not in ['imputer.pkl', 'scaler.pkl', 'feature_names.pkl', 'train_test_split.pkl']:
                model_path = os.path.join(models_dir, model_file)
                print(f"Trying alternative model: {model_file}")
                break
        else:
            print("No suitable model files found.")
            return
    
    model = joblib.load(model_path)
    print(f"Model loaded: {type(model).__name__}")
    
    # Examine model attributes
    print("\nModel attributes:")
    if hasattr(model, 'feature_names_in_'):
        print(f"feature_names_in_: {model.feature_names_in_}")
    if hasattr(model, 'n_features_in_'):
        print(f"n_features_in_: {model.n_features_in_}")
    
    # Load feature names
    feature_path = os.path.join(models_dir, 'feature_names.pkl')
    if os.path.exists(feature_path):
        feature_names = joblib.load(feature_path)
        print(f"\nFeature names from file: {feature_names}")
        
        # Check if 'Outcome' is in feature_names
        if 'Outcome' in feature_names:
            print("WARNING: 'Outcome' is included in feature_names!")
    else:
        print(f"Feature names file not found at {feature_path}")
    
    # Load the original dataset to see its structure
    try:
        data_path = os.path.join(project_dir, 'data', 'diabetes.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            print(f"\nOriginal dataset columns: {data.columns.tolist()}")
        else:
            print(f"Original dataset not found at {data_path}")
    except Exception as e:
        print(f"Error loading original dataset: {e}")
    
    # Create test patient data
    test_patient = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    print(f"\nTest patient data: {test_patient}")
    
    # Try different approaches for prediction
    print("\nTrying different prediction approaches:")
    
    # Approach 1: Direct prediction with model's feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        try:
            # Create DataFrame with exactly the features the model expects
            model_features = model.feature_names_in_
            X = pd.DataFrame(columns=model_features)
            
            # Fill in the basic features we have
            for col in test_patient.keys():
                if col in model_features:
                    X[col] = [test_patient[col]]
            
            # Fill in any derived features
            # Glucose_to_BMI_Ratio
            if 'Glucose_to_BMI_Ratio' in model_features:
                X['Glucose_to_BMI_Ratio'] = test_patient['Glucose'] / test_patient['BMI']
            
            # BMI_to_Age_Ratio
            if 'BMI_to_Age_Ratio' in model_features:
                X['BMI_to_Age_Ratio'] = test_patient['BMI'] / test_patient['Age']
            
            # Glucose_to_Insulin_Ratio
            if 'Glucose_to_Insulin_Ratio' in model_features:
                X['Glucose_to_Insulin_Ratio'] = test_patient['Glucose'] / (test_patient['Insulin'] + 1)
            
            # Age_BMI_Interaction
            if 'Age_BMI_Interaction' in model_features:
                X['Age_BMI_Interaction'] = test_patient['Age'] * test_patient['BMI'] / 100
            
            # Glucose_BMI_Interaction
            if 'Glucose_BMI_Interaction' in model_features:
                X['Glucose_BMI_Interaction'] = test_patient['Glucose'] * test_patient['BMI'] / 100
            
            # BMI Category features
            bmi = test_patient['BMI']
            if 'BMI_Category_Normal' in model_features:
                X['BMI_Category_Normal'] = 1 if 18.5 <= bmi < 25 else 0
            if 'BMI_Category_Overweight' in model_features:
                X['BMI_Category_Overweight'] = 1 if 25 <= bmi < 30 else 0
            if 'BMI_Category_Obese_I' in model_features:
                X['BMI_Category_Obese_I'] = 1 if 30 <= bmi < 35 else 0
            if 'BMI_Category_Obese_II_III' in model_features:
                X['BMI_Category_Obese_II_III'] = 1 if bmi >= 35 else 0
            
            # Age Group features
            age = test_patient['Age']
            if 'Age_Group_Young' in model_features:
                X['Age_Group_Young'] = 1 if age < 30 else 0
            if 'Age_Group_Middle_Aged' in model_features:
                X['Age_Group_Middle_Aged'] = 1 if 30 <= age < 45 else 0
            if 'Age_Group_Senior' in model_features:
                X['Age_Group_Senior'] = 1 if 45 <= age < 60 else 0
            
            # Add Outcome if needed
            if 'Outcome' in model_features:
                X['Outcome'] = 0
            
            # Fill any missing columns with 0
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
            
            # Ensure columns are in the right order
            X = X[model_features]
            
            print(f"Created input data with shape: {X.shape}")
            print(f"Input data columns: {X.columns.tolist()}")
            
            # Make prediction
            proba = model.predict_proba(X)[0, 1]
            print(f"Prediction probability: {proba:.4f}")
            
            # Categorize risk
            if proba < 0.3:
                risk = "Low"
            elif proba < 0.7:
                risk = "Moderate"
            else:
                risk = "High"
            print(f"Risk category: {risk}")
            
        except Exception as e:
            print(f"Error in Approach 1: {e}")
    
    # Approach 2: Try with raw numpy array
    try:
        # If we know the exact number of features the model expects
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            X_array = np.zeros((1, n_features))
            
            # Fill in whatever values we can
            for i, col in enumerate(model.feature_names_in_):
                if col in test_patient:
                    X_array[0, i] = test_patient[col]
            
            # Make prediction
            proba = model.predict_proba(X_array)[0, 1]
            print(f"\nApproach 2 prediction probability: {proba:.4f}")
            
            # Categorize risk
            if proba < 0.3:
                risk = "Low"
            elif proba < 0.7:
                risk = "Moderate"
            else:
                risk = "High"
            print(f"Risk category: {risk}")
    except Exception as e:
        print(f"Error in Approach 2: {e}")
    
    print("\nDebug complete.")

if __name__ == "__main__":
    debug_prediction()