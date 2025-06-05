import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


def load_model_and_preprocessors(models_dir, model_name=None):
    """
    Load the trained model and preprocessors.
    
    Parameters:
    -----------
    models_dir: str
        Directory containing the trained models and preprocessors
    model_name: str, optional
        Name of the model to load (default: best model based on evaluation)
        
    Returns:
    --------
    model: trained model
        The loaded machine learning model
    imputer: fitted imputer
        For handling missing values
    scaler: fitted scaler
        For standardizing features
    feature_names: list
        List of feature names expected by the model
    model_name: str
        Name of the loaded model
    """
    # Check if models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load feature names
    feature_path = os.path.join(models_dir, "feature_names.pkl")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_path}")
    feature_names = joblib.load(feature_path)
    
    # Remove 'Outcome' from feature_names if present
    if 'Outcome' in feature_names:
        feature_names = [f for f in feature_names if f != 'Outcome']
    
    # Load preprocessors
    imputer_path = os.path.join(models_dir, 'imputer.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    if not os.path.exists(imputer_path):
        raise FileNotFoundError(f"Imputer file not found: {imputer_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    
    # If model_name is not provided, load the best model based on evaluation results
    best_model_name = None  # Initialize the variable properly
    
    if model_name is None:
        # Try to load evaluation results
        try:
            results_path = os.path.join(os.path.dirname(models_dir), 'model_evaluation_results.csv')
            if os.path.exists(results_path):
                results = pd.read_csv(results_path)
                
                # Get the model with highest ROC-AUC
                max_auc_idx = results['ROC-AUC'].idxmax()
                
                # Get the model name from the index or from a column
                if 'Unnamed: 0' in results.columns:
                    # Model names are likely in the first column
                    best_model_name = str(results.loc[max_auc_idx, 'Unnamed: 0'])
                else:
                    # Try to get model name from index
                    try:
                        best_model_name = str(results.index[max_auc_idx])
                    except:
                        # Default to gradient boosting if we can't determine the model name
                        print("Warning: Could not determine model name from results index")
                        best_model_name = 'gradient_boosting'
                
                # Convert model name to proper format
                best_model_name = best_model_name.replace(' ', '_').lower()
            else:
                # Default to gradient boosting if results not available
                best_model_name = 'gradient_boosting'
        except Exception as e:
            print(f"Warning: Could not determine best model from results: {e}")
            best_model_name = 'gradient_boosting'
    else:
        best_model_name = model_name.replace(' ', '_').lower()
    
    # Load the model
    model_path = os.path.join(models_dir, f"{best_model_name}.pkl")
    if not os.path.exists(model_path):
        # Try alternatives if the specific model isn't found
        available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and f != 'imputer.pkl' 
                           and f != 'scaler.pkl' and f != 'feature_names.pkl' and f != 'train_test_split.pkl']
        
        if available_models:
            print(f"Warning: Model {best_model_name}.pkl not found. Using {available_models[0]} instead.")
            model_path = os.path.join(models_dir, available_models[0])
            best_model_name = os.path.splitext(available_models[0])[0]
        else:
            raise FileNotFoundError(f"No model files found in {models_dir}")
    
    model = joblib.load(model_path)
    
    return model, imputer, scaler, feature_names, best_model_name


def preprocess_patient_data(patient_data, imputer, scaler, feature_names):
    """
    Preprocess patient data for prediction.
    
    Parameters:
    -----------
    patient_data: dict
        Dictionary with patient features
    imputer: fitted imputer
        For handling missing values
    scaler: fitted scaler
        For standardizing features
    feature_names: list
        List of feature names expected by the model
        
    Returns:
    --------
    patient_scaled: numpy array
        Preprocessed and scaled patient data ready for prediction
    patient_features: pandas DataFrame
        Preprocessed patient data before scaling
    """
    # Remove 'Outcome' from feature_names if present
    if 'Outcome' in feature_names:
        feature_names = [f for f in feature_names if f != 'Outcome']
        
    # Define columns where zero is not a valid value
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Validate input data
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    missing_columns = [col for col in required_columns if col not in patient_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Replace zeros with NaN for columns where zero is not valid
    for column in zero_columns:
        if column in patient_df.columns:
            patient_df[column] = patient_df[column].replace(0, np.nan)
    
    # Impute missing values
    patient_imputed = pd.DataFrame(
        imputer.transform(patient_df),
        columns=patient_df.columns
    )
    
    # Create engineered features
    if 'Glucose_to_BMI_Ratio' in feature_names:
        patient_imputed['Glucose_to_BMI_Ratio'] = patient_imputed['Glucose'] / patient_imputed['BMI']
    
    if 'Age_BMI_Interaction' in feature_names:
        patient_imputed['Age_BMI_Interaction'] = patient_imputed['Age'] * patient_imputed['BMI'] / 100
    
    if 'Insulin_Log' in feature_names:
        patient_imputed['Insulin_Log'] = np.log1p(patient_imputed['Insulin'])
    
    if 'DiabetesPedigreeFunction_Log' in feature_names:
        patient_imputed['DiabetesPedigreeFunction_Log'] = np.log1p(patient_imputed['DiabetesPedigreeFunction'])
    
    # Create BMI category
    if any('BMI_Category' in feature for feature in feature_names):
        bins = [0, 18.5, 25, 30, 35, 100]
        labels = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II_III']
        patient_imputed['BMI_Category'] = pd.cut(patient_imputed['BMI'], bins=bins, labels=labels)
        
        # Create dummy variables
        bmi_dummies = pd.get_dummies(patient_imputed['BMI_Category'], prefix='BMI_Category')
        patient_imputed = pd.concat([patient_imputed, bmi_dummies], axis=1)
    
    # Create age group
    if any('Age_Group' in feature for feature in feature_names):
        age_bins = [0, 30, 45, 60, 100]
        age_labels = ['Young', 'Middle_Aged', 'Senior', 'Elderly']
        patient_imputed['Age_Group'] = pd.cut(patient_imputed['Age'], bins=age_bins, labels=age_labels)
        
        # Create dummy variables
        age_dummies = pd.get_dummies(patient_imputed['Age_Group'], prefix='Age_Group')
        patient_imputed = pd.concat([patient_imputed, age_dummies], axis=1)
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(patient_imputed.columns)
    for feature in missing_features:
        patient_imputed[feature] = 0
    
    # Select only the features used by the model
    patient_features = patient_imputed[feature_names]
    
    # Scale features
    patient_scaled = scaler.transform(patient_features)
    
    return patient_scaled, patient_features


def predict_diabetes_risk(patient_data, model, imputer, scaler, feature_names, save_dir=None):
    """
    Predict diabetes risk for a patient using the trained model.
    
    Parameters:
    -----------
    patient_data: dict
        Dictionary with patient features
    model: trained model
        The machine learning model
    imputer: fitted imputer
        For handling missing values
    scaler: fitted scaler
        For standardizing features
    feature_names: list
        List of feature names expected by the model (not used directly)
    save_dir: str, optional
        Directory to save visualization files
    
    Returns:
    --------
    dict
        Dictionary containing risk probability, category, and contributing factors
    """
    # Debug info
    print("Debug: Checking model features")
    if hasattr(model, 'feature_names_in_'):
        print(f"Model expects features: {model.feature_names_in_}")
        # Use the model's feature_names_in_ as the source of truth
        model_features = list(model.feature_names_in_)
    else:
        # Fallback to provided feature_names if model doesn't have feature_names_in_
        model_features = feature_names
        print(f"Using provided feature names: {model_features}")
    
    # First, preprocess the basic patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Handle missing or zero values
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        if column in patient_df.columns:
            patient_df[column] = patient_df[column].replace(0, np.nan)
    
    # Apply imputation if there are missing values
    if patient_df.isna().any().any():
        patient_imputed = pd.DataFrame(
            imputer.transform(patient_df),
            columns=patient_df.columns
        )
    else:
        patient_imputed = patient_df.copy()
    
    # Now construct the full feature set directly from the model's expected features
    final_input = pd.DataFrame(index=[0])
    
    # Add basic features from patient data
    for col in patient_imputed.columns:
        if col in model_features:
            final_input[col] = patient_imputed[col].values
    
    # Generate derived features based on the model's expected features
    # BMI_to_Age_Ratio
    if 'BMI_to_Age_Ratio' in model_features:
        final_input['BMI_to_Age_Ratio'] = patient_imputed['BMI'] / patient_imputed['Age']
    
    # Glucose_to_Insulin_Ratio
    if 'Glucose_to_Insulin_Ratio' in model_features:
        final_input['Glucose_to_Insulin_Ratio'] = patient_imputed['Glucose'] / (patient_imputed['Insulin'] + 1)
    
    # Age_BMI_Interaction
    if 'Age_BMI_Interaction' in model_features:
        final_input['Age_BMI_Interaction'] = patient_imputed['Age'] * patient_imputed['BMI'] / 100
    
    # Glucose_BMI_Interaction
    if 'Glucose_BMI_Interaction' in model_features:
        final_input['Glucose_BMI_Interaction'] = patient_imputed['Glucose'] * patient_imputed['BMI'] / 100
    
    # BMI Category features
    bmi = patient_imputed['BMI'].values[0]
    if 'BMI_Category_Normal' in model_features:
        final_input['BMI_Category_Normal'] = 1 if 18.5 <= bmi < 25 else 0
    if 'BMI_Category_Overweight' in model_features:
        final_input['BMI_Category_Overweight'] = 1 if 25 <= bmi < 30 else 0
    if 'BMI_Category_Obese_I' in model_features:
        final_input['BMI_Category_Obese_I'] = 1 if 30 <= bmi < 35 else 0
    if 'BMI_Category_Obese_II_III' in model_features:
        final_input['BMI_Category_Obese_II_III'] = 1 if bmi >= 35 else 0
    
    # Age Group features
    age = patient_imputed['Age'].values[0]
    if 'Age_Group_Young' in model_features:
        final_input['Age_Group_Young'] = 1 if age < 30 else 0
    if 'Age_Group_Middle_Aged' in model_features:
        final_input['Age_Group_Middle_Aged'] = 1 if 30 <= age < 45 else 0
    if 'Age_Group_Senior' in model_features:
        final_input['Age_Group_Senior'] = 1 if 45 <= age < 60 else 0
    if 'Age_Group_Elderly' in model_features:
        final_input['Age_Group_Elderly'] = 1 if age >= 60 else 0
    
    # Fill any remaining missing features with 0
    for feature in model_features:
        if feature not in final_input.columns:
            final_input[feature] = 0
    
    # Make sure columns are in the exact order the model expects
    final_input = final_input[model_features]
    
    # Apply scaling if needed
    if scaler is not None:
        try:
            final_input_scaled = pd.DataFrame(
                scaler.transform(final_input),
                columns=final_input.columns
            )
        except Exception as e:
            print(f"Warning: Scaling failed: {e}. Using unscaled features.")
            final_input_scaled = final_input
    else:
        final_input_scaled = final_input
    
    # Make prediction
    try:
        risk_probability = model.predict_proba(final_input_scaled)[0, 1]
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Last resort - try adding Outcome
        if 'Outcome' not in final_input_scaled.columns:
            final_input_scaled['Outcome'] = 0
            try:
                risk_probability = model.predict_proba(final_input_scaled)[0, 1]
            except Exception as e2:
                print(f"Failed even with Outcome added: {e2}")
                risk_probability = 0.5  # Default fallback
    
    # Categorize risk
    if risk_probability < 0.3:
        risk_category = "Low"
    elif risk_probability < 0.7:
        risk_category = "Moderate"
    else:
        risk_category = "High"
    
    # For risk factors, use the feature importances from the model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create a mapping of feature to importance
        feature_importance = {feature: importance for feature, importance in zip(model_features, importances)}
        
        # Sort by importance and take top 5
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        risk_factors = [
            {
                "feature": feature,
                "importance": importance,
                "value": final_input[feature].values[0]
            }
            for feature, importance in top_features
        ]
    else:
        # Fallback for non-tree models
        risk_factors = [
            {"feature": "Glucose", "value": patient_imputed["Glucose"].values[0]},
            {"feature": "BMI", "value": patient_imputed["BMI"].values[0]},
            {"feature": "Age", "value": patient_imputed["Age"].values[0]}
        ]
    
    # Generate visualizations if save_dir is provided
    if save_dir and hasattr(model, 'feature_importances_'):
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Model feature importance plot
            plt.figure(figsize=(10, 6))
            importances_series = pd.Series(importances, index=model_features)
            top_importances = importances_series.nlargest(10)
            sns.barplot(x=top_importances.values, y=top_importances.index)
            plt.title('Top 10 Model Features')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'model_feature_importance.png'))
            plt.close()
            
            # Patient-specific contribution plot
            plt.figure(figsize=(10, 6))
            patient_values = final_input.iloc[0]
            contributions = {feature: importance * abs(patient_values[feature]) for feature, importance in zip(model_features, importances)}
            contribution_series = pd.Series(contributions)
            top_contributions = contribution_series.nlargest(10)
            sns.barplot(x=top_contributions.values, y=top_contributions.index)
            plt.title('Top 10 Feature Contributions for This Patient')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'patient_feature_contributions.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    result = {
        "risk_probability": float(risk_probability),
        "risk_category": risk_category,
        "risk_factors": risk_factors
    }
    
    return result

def generate_risk_report(patient_data, prediction_result, save_dir=None):
    """
    Generate a risk assessment report for the patient.
    
    Parameters:
    -----------
    patient_data: dict
        Dictionary with patient features
    prediction_result: dict
        Result from predict_diabetes_risk function
    save_dir: str, optional
        Directory to save the report
        
    Returns:
    --------
    str
        Formatted risk assessment report
    """
    report = [
        "Diabetes Risk Assessment Report",
        "==============================\n",
        f"Risk Probability: {prediction_result['risk_probability']:.2f}",
        f"Risk Category: {prediction_result['risk_category']}",
        "\nRisk Factors:"
    ]
    
    for factor in prediction_result['risk_factors']:
        if 'importance' in factor:
            report.append(f"- {factor['feature']}: Value = {factor['value']:.2f}, Importance = {factor['importance']:.4f}")
        else:
            report.append(f"- {factor['feature']}: Value = {factor['value']}")
    
    report.extend([
        "\nPatient Data:",
        f"- Age: {patient_data.get('Age', 'N/A')}",
        f"- BMI: {patient_data.get('BMI', 'N/A'):.1f}",
        f"- Glucose: {patient_data.get('Glucose', 'N/A')} mg/dL",
        f"- Blood Pressure: {patient_data.get('BloodPressure', 'N/A')} mm Hg",
        f"- Pregnancies: {patient_data.get('Pregnancies', 'N/A')}",
        f"- Diabetes Pedigree Function: {patient_data.get('DiabetesPedigreeFunction', 'N/A'):.3f}"
    ])
    
    # Add recommendations based on risk category
    report.extend([
        "\nRecommendations:",
    ])
    
    if prediction_result['risk_category'] == "Low":
        report.extend([
            "- Continue with standard health maintenance",
            "- Follow healthy diet and exercise guidelines",
            "- Get regular check-ups every 1-3 years"
        ])
    elif prediction_result['risk_category'] == "Moderate":
        report.extend([
            "- Consult with healthcare provider within 3-6 months",
            "- Consider lifestyle modifications (diet, exercise)",
            "- Monitor blood glucose levels periodically",
            "- Follow up with healthcare provider annually"
        ])
    else:  # High risk
        report.extend([
            "- Schedule appointment with healthcare provider promptly",
            "- Consider comprehensive diabetes screening",
            "- Implement intensive lifestyle modifications",
            "- Follow up with healthcare provider every 3-6 months",
            "- Consider preventive interventions as recommended by healthcare provider"
        ])
    
    report_text = "\n".join(report)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'risk_assessment_report.txt'), 'w') as f:
            f.write(report_text)
    
    return report_text

def batch_predict(data_path, models_dir, save_dir=None):
    """
    Make predictions for multiple patients from a CSV file.
    
    Parameters:
    -----------
    data_path: str
        Path to the CSV file containing patient data
    models_dir: str
        Directory containing the trained model and preprocessors
    save_dir: str, optional
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing predictions for each patient
    """
    # Load patient data
    try:
        patients_df = pd.read_csv(data_path)
        print(f"Loaded {len(patients_df)} patients from {data_path}")
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return None
    
    # Load model and preprocessors
    try:
        model, imputer, scaler, feature_names, model_name = load_model_and_preprocessors(models_dir)
        print(f"Using model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Make predictions for each patient
    results = []
    for i, row in patients_df.iterrows():
        patient_data = row.to_dict()
        
        try:
            # Preprocess patient data
            patient_scaled, patient_features = preprocess_patient_data(
                patient_data, imputer, scaler, feature_names
            )
            
            # Predict probability
            risk_probability = model.predict_proba(patient_scaled)[0, 1]
            
            # Categorize risk
            if risk_probability < 0.3:
                risk_category = "Low"
            elif risk_probability < 0.7:
                risk_category = "Moderate"
            else:
                risk_category = "High"
            
            # Add prediction to results
            results.append({
                'patient_id': i,
                'risk_probability': risk_probability,
                'risk_category': risk_category
            })
        except Exception as e:
            print(f"Error predicting for patient {i}: {e}")
            results.append({
                'patient_id': i,
                'risk_probability': None,
                'risk_category': 'Error'
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, 'batch_predictions.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Batch predictions saved to {results_path}")
        
        # Create summary visualization
        plt.figure(figsize=(10, 6))
        summary = results_df['risk_category'].value_counts()
        colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Error': 'gray'}
        summary.plot(kind='bar', color=[colors.get(cat, 'blue') for cat in summary.index])
        plt.title('Risk Category Distribution')
        plt.xlabel('Risk Category')
        plt.ylabel('Number of Patients')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'risk_distribution.png'))
        plt.close()
        
        # Create summary report
        summary_text = [
            "Diabetes Risk Prediction - Batch Summary",
            "=======================================\n",
            f"Total patients analyzed: {len(results_df)}",
            "\nRisk Category Distribution:"
        ]
        
        for category, count in summary.items():
            summary_text.append(f"- {category}: {count} patients ({count/len(results_df)*100:.1f}%)")
        
        with open(os.path.join(save_dir, 'batch_summary.txt'), 'w') as f:
            f.write("\n".join(summary_text))
    
    return results_df

def validate_numeric_input(prompt, min_value=0, max_value=float('inf'), allow_zero=True):
    """
    Validate numeric input from the user.
    
    Parameters:
    -----------
    prompt: str
        Prompt to display to the user
    min_value: float
        Minimum allowed value
    max_value: float
        Maximum allowed value
    allow_zero: bool
        Whether to allow zero as a valid value
    
    Returns:
    --------
    float
        Validated input value
    """
    while True:
        try:
            value = float(input(prompt))
            if not allow_zero and value == 0:
                print(f"Error: Value cannot be zero. Please enter a value between {min_value} and {max_value}.")
                continue
            if value < min_value or value > max_value:
                print(f"Error: Value must be between {min_value} and {max_value}.")
                continue
            return value
        except ValueError:
            print("Error: Please enter a valid number.")

def get_patient_data_from_user():
    """
    Get patient data from user input with validation.
    
    Returns:
    --------
    dict
        Dictionary with patient features
    """
    print("\nEnter patient data for diabetes risk prediction:")
    
    patient_data = {}
    
    # Get pregnancies (integer, non-negative)
    patient_data['Pregnancies'] = int(validate_numeric_input(
        "Number of pregnancies: ", min_value=0, max_value=20
    ))
    
    # Get glucose (positive, typical range 70-200)
    patient_data['Glucose'] = validate_numeric_input(
        "Plasma glucose concentration (mg/dL): ", min_value=0, max_value=500, allow_zero=False
    )
    
    # Get blood pressure (positive, typical range 60-130)
    patient_data['BloodPressure'] = validate_numeric_input(
        "Diastolic blood pressure (mm Hg): ", min_value=0, max_value=250
    )
    
    # Get skin thickness (non-negative, typical range 5-50)
    patient_data['SkinThickness'] = validate_numeric_input(
        "Triceps skin fold thickness (mm): ", min_value=0, max_value=100
    )
    
    # Get insulin (non-negative, typical range 0-300)
    patient_data['Insulin'] = validate_numeric_input(
        "2-Hour serum insulin (mu U/ml): ", min_value=0, max_value=1000
    )
    
    # Get BMI (positive, typical range 15-50)
    patient_data['BMI'] = validate_numeric_input(
        "Body mass index: ", min_value=10, max_value=70, allow_zero=False
    )
    
    # Get diabetes pedigree function (positive, typical range 0.1-2.5)
    patient_data['DiabetesPedigreeFunction'] = validate_numeric_input(
        "Diabetes pedigree function: ", min_value=0, max_value=5
    )
    
    # Get age (positive integer)
    patient_data['Age'] = int(validate_numeric_input(
        "Age (years): ", min_value=1, max_value=120
    ))
    
    return patient_data

def display_help():
    """Display help information for the script."""
    help_text = """
Diabetes Risk Prediction Tool
============================

This tool predicts the risk of diabetes based on health parameters.

Usage:
  python prediction.py [OPTIONS]

Options:
  --mode MODE       Prediction mode: 'single' for one patient, 'batch' for multiple patients
                    from a CSV file, or 'help' to display this help message.
  --data FILE       Path to CSV file containing patient data (required for batch mode).
  --model MODEL     Name of the model to use for prediction (optional).
  --output DIR      Directory to save results (optional).

Examples:
  python prediction.py                          # Interactive mode for a single patient
  python prediction.py --mode batch --data patients.csv  # Batch prediction from CSV file
  python prediction.py --help                   # Display this help message

Input CSV Format:
  The CSV file should have the following columns:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age

For more information, see the documentation.
"""
    print(help_text)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Diabetes Risk Prediction')
    parser.add_argument('--mode', choices=['single', 'batch', 'help'], default='single',
                        help='Prediction mode: single patient, batch from CSV, or help')
    parser.add_argument('--data', type=str, help='Path to CSV file for batch prediction')
    parser.add_argument('--model', type=str, help='Name of the model to use (optional)')
    parser.add_argument('--output', type=str, help='Directory to save results (optional)')
    args = parser.parse_args()
    
    # Display help if requested
    if args.mode == 'help':
        display_help()
        sys.exit(0)
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir.endswith('src'):
        base_dir = os.path.dirname(base_dir)
    
    models_dir = args.output if args.output else os.path.join(base_dir, 'results', 'models')
    results_dir = os.path.dirname(models_dir) if os.path.basename(models_dir) == 'models' else models_dir
    patient_analysis_dir = os.path.join(results_dir, 'patient_analysis')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(patient_analysis_dir, exist_ok=True)
    
    # Check if models directory exists and contains required files
    try:
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        required_files = ['feature_names.pkl', 'imputer.pkl', 'scaler.pkl']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(models_dir, f))]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files in models directory: {', '.join(missing_files)}")
        
        # Check if at least one model file exists
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') 
                      and f not in ['imputer.pkl', 'scaler.pkl', 'feature_names.pkl', 'train_test_split.pkl']]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease make sure you have trained the models first by running the training pipeline.")
        print("You can do this by running the main.py file with the --mode train argument.")
        sys.exit(1)
    
    if args.mode == 'batch':
        # Validate that data file is provided for batch mode
        if not args.data:
            print("Error: --data argument is required for batch mode")
            print("Use 'python prediction.py --help' for more information")
            sys.exit(1)
        
        # Run batch prediction
        print(f"\nRunning batch prediction on {args.data}...")
        batch_output_dir = os.path.join(results_dir, 'batch_results')
        results_df = batch_predict(
            args.data, models_dir, save_dir=batch_output_dir
        )
        
        if results_df is not None:
            print("\nBatch prediction completed successfully.")
            print("\nRisk Category Summary:")
            summary = results_df['risk_category'].value_counts()
            for category, count in summary.items():
                print(f"- {category}: {count} patients ({count/len(results_df)*100:.1f}%)")
            print(f"\nResults saved to {batch_output_dir}")
    else:  # single mode
        # Single patient prediction
        try:
            # Load model and preprocessors
            model, imputer, scaler, feature_names, model_name = load_model_and_preprocessors(
                models_dir, model_name=args.model
            )
            
            print(f"\nUsing model: {model_name}")
            
            # Get patient data from user
            patient_data = get_patient_data_from_user()
            
            # Confirm the input data
            print("\nConfirm input data:")
            for key, value in patient_data.items():
                print(f"- {key}: {value}")
            
            confirm = input("\nIs this data correct? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Prediction cancelled. Please run the script again with correct data.")
                sys.exit(0)
            
            # Predict risk
            print("\nCalculating diabetes risk...")
            prediction_result = predict_diabetes_risk(
                patient_data, model, imputer, scaler, feature_names,
                save_dir=patient_analysis_dir
            )
            
            # Generate report
            report = generate_risk_report(
                patient_data, prediction_result,
                save_dir=patient_analysis_dir
            )
            
            print("\nDiabetes Risk Prediction:")
            print(report)
            
            print("\nAnalysis files saved to:", patient_analysis_dir)
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print("\nPlease check your input data and try again.")
            sys.exit(1)