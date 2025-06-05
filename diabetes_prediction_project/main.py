import os
import joblib
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_and_save_models
from src.model_evaluation import evaluate_and_analyze_models, evaluate_models
from src.prediction import load_model_and_preprocessors, predict_diabetes_risk, generate_risk_report

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data',
        'results',
        'results/figures',
        'results/models',
        'results/patient_analysis',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return {dir_name: os.path.join(os.getcwd(), dir_name) for dir_name in directories}

def train_pipeline(data_path, dirs):
    """Run the complete enhanced training pipeline."""
    print("\n" + "="*50)
    print("ADVANCED DIABETES PREDICTION MODEL - TRAINING PIPELINE")
    print("="*50)
    
    print("\n=== Step 1: Advanced Data Preprocessing ===")
    data_imputed, imputer = preprocess_data(data_path, dirs['results'])
    
    print("\n=== Step 2: Advanced Feature Engineering ===")
    X_scaled, y, feature_names, scaler = engineer_features(data_imputed, dirs['results'])
    
    print("\n=== Step 3: Advanced Model Training ===")
    X_train, X_test, y_train, y_test, models = train_and_save_models(
        X_scaled, y, feature_names, dirs['results/models']
    )
    
    # Save preprocessors
    print("\nSaving preprocessors...")
    joblib.dump(imputer, os.path.join(dirs['results/models'], 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(dirs['results/models'], 'scaler.pkl'))
    
    print("\n=== Step 4: Advanced Model Evaluation ===")
    results_df = evaluate_models(models, X_test, y_test)
    print("\nModel Evaluation Results:")
    print(results_df)
    
    # Save evaluation results
    results_df.to_csv(os.path.join(dirs['results'], 'model_evaluation_results.csv'))
    
    # Find best model based on ROC-AUC
    best_model_name = results_df['ROC-AUC'].idxmax()
    best_model = models[best_model_name]
    
    print("\n" + "="*50)
    print(f"Training pipeline completed successfully!")
    print(f"Best model: {best_model_name} with ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
    print("="*50)
    
    return best_model_name

def predict_for_patient(patient_data, dirs):
    """Make prediction for a single patient."""
    print("\n" + "="*50)
    print("DIABETES RISK PREDICTION")
    print("="*50)
    
    # Load model and preprocessors
    try:
        model, imputer, scaler, feature_names, model_name = load_model_and_preprocessors(dirs['results/models'])
        print(f"\nUsing model: {model_name}")
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease make sure you have trained the models first by running:")
        print("python main.py --mode train")
        return None
    
    # Predict risk
    try:
        prediction_result = predict_diabetes_risk(
            patient_data, model, imputer, scaler, feature_names,
            save_dir=dirs['results/patient_analysis']
        )
        
        # Generate report
        report = generate_risk_report(
            patient_data, prediction_result,
            save_dir=dirs['results/patient_analysis']
        )
        
        print("\nDiabetes Risk Assessment Report:")
        print(report)
        
        print("\nAnalysis files saved to:", dirs['results/patient_analysis'])
        return prediction_result
    except Exception as e:
        print(f"\nError during prediction: {e}")
        return None

def get_patient_data():
    """Get patient data from user input."""
    print("\nEnter patient data for diabetes risk prediction:")
    
    try:
        patient_data = {
            'Pregnancies': int(input("Number of pregnancies: ")),
            'Glucose': float(input("Plasma glucose concentration (mg/dL): ")),
            'BloodPressure': float(input("Diastolic blood pressure (mm Hg): ")),
            'SkinThickness': float(input("Triceps skin fold thickness (mm): ")),
            'Insulin': float(input("2-Hour serum insulin (mu U/ml): ")),
            'BMI': float(input("Body mass index: ")),
            'DiabetesPedigreeFunction': float(input("Diabetes pedigree function: ")),
            'Age': int(input("Age (years): "))
        }
        
        # Confirm data
        print("\nConfirm input data:")
        for key, value in patient_data.items():
            print(f"- {key}: {value}")
        
        confirm = input("\nIs this data correct? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Input cancelled. Please try again.")
            return None
        
        return patient_data
    except ValueError:
        print("Error: Please enter numeric values.")
        return None

def batch_predict(data_path, dirs):
    """Make predictions for multiple patients from a CSV file."""
    print("\n" + "="*50)
    print("DIABETES RISK PREDICTION - BATCH MODE")
    print("="*50)
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        return None
    
    # Load model and preprocessors
    try:
        model, imputer, scaler, feature_names, model_name = load_model_and_preprocessors(dirs['results/models'])
        print(f"\nUsing model: {model_name}")
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease make sure you have trained the models first by running:")
        print("python main.py --mode train")
        return None
    
    # Load patient data
    try:
        patients_df = pd.read_csv(data_path)
        print(f"Loaded {len(patients_df)} patients from {data_path}")
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return None
    
    # Check required columns
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    missing_columns = [col for col in required_columns if col not in patients_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        return None
    
    # Create batch results directory
    batch_dir = os.path.join(dirs['results'], 'batch_results')
    os.makedirs(batch_dir, exist_ok=True)
    
    # Make predictions for each patient
    results = []
    print("\nProcessing patients...")
    
    for i, row in patients_df.iterrows():
        try:
            patient_data = row.to_dict()
            
            # Preprocess and predict
            prediction_result = predict_diabetes_risk(
                patient_data, model, imputer, scaler, feature_names
            )
            
            # Add prediction to results
            results.append({
                'patient_id': i,
                'risk_probability': prediction_result['risk_probability'],
                'risk_category': prediction_result['risk_category']
            })
            
            # Print progress for every 10th patient
            if (i + 1) % 10 == 0 or i == len(patients_df) - 1:
                print(f"Processed {i + 1}/{len(patients_df)} patients")
                
        except Exception as e:
            print(f"Error predicting for patient {i}: {e}")
            results.append({
                'patient_id': i,
                'risk_probability': None,
                'risk_category': 'Error'
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(batch_dir, 'batch_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nBatch predictions saved to {results_path}")
    
    # Create summary visualization
    summary = results_df['risk_category'].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Error': 'gray'}
    summary.plot(kind='bar', color=[colors.get(cat, 'blue') for cat in summary.index])
    plt.title('Risk Category Distribution')
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Patients')
    plt.tight_layout()
    plt.savefig(os.path.join(batch_dir, 'risk_distribution.png'))
    
    # Print summary
    print("\nRisk Category Summary:")
    for category, count in summary.items():
        print(f"- {category}: {count} patients ({count/len(results_df)*100:.1f}%)")
    
    # Create summary report
    summary_text = [
        "Diabetes Risk Prediction - Batch Summary",
        "=======================================\n",
        f"Total patients analyzed: {len(results_df)}",
        "\nRisk Category Distribution:"
    ]
    
    for category, count in summary.items():
        summary_text.append(f"- {category}: {count} patients ({count/len(results_df)*100:.1f}%)")
    
    with open(os.path.join(batch_dir, 'batch_summary.txt'), 'w') as f:
        f.write("\n".join(summary_text))
    
    print(f"\nSummary report saved to {os.path.join(batch_dir, 'batch_summary.txt')}")
    
    return results_df

def display_help():
    """Display help information."""
    help_text = """
Diabetes Risk Prediction Tool
============================

This tool predicts the risk of diabetes based on health parameters.

Usage:
  python main.py [OPTIONS]

Options:
  --mode MODE       Prediction mode: 'train' to train models, 'predict' for single patient,
                    'batch' for multiple patients from a CSV file, or 'help' to display this message.
  --data FILE       Path to CSV file (required for training and batch modes).
  --output DIR      Directory to save results (optional).

Examples:
  python main.py --mode train --data data/diabetes.csv       # Train models
  python main.py --mode predict                              # Predict for a single patient
  python main.py --mode batch --data data/patients.csv       # Batch prediction
  python main.py --help                                      # Display this help message

For more information, see the documentation.
"""
    print(help_text)

def main():
    """Main function to run the diabetes prediction system."""
    parser = argparse.ArgumentParser(description='Diabetes Risk Prediction System')
    parser.add_argument('--mode', choices=['train', 'predict', 'batch', 'help'], default='train',
                        help='Operation mode: train, predict, batch, or help')
    parser.add_argument('--data', type=str, default='data/diabetes.csv',
                        help='Path to data file (CSV)')
    parser.add_argument('--output', type=str, help='Directory to save results (optional)')
    args = parser.parse_args()
    
    # Setup directories
    dirs = setup_directories()
    
    # Override results directory if specified
    if args.output:
        dirs['results'] = args.output
        dirs['results/figures'] = os.path.join(args.output, 'figures')
        dirs['results/models'] = os.path.join(args.output, 'models')
        dirs['results/patient_analysis'] = os.path.join(args.output, 'patient_analysis')
        
        for directory in [dirs['results'], dirs['results/figures'], dirs['results/models'], dirs['results/patient_analysis']]:
            os.makedirs(directory, exist_ok=True)
    
    # Display help
    if args.mode == 'help':
        display_help()
        return
    
    # Execute requested mode
    if args.mode == 'train':
        train_pipeline(args.data, dirs)
    
    elif args.mode == 'predict':
        patient_data = get_patient_data()
        if patient_data:
            predict_for_patient(patient_data, dirs)
    
    elif args.mode == 'batch':
        batch_predict(args.data, dirs)

if __name__ == "__main__":
    main()