# Sample Dataset Structure for Blood Pressure Analysis
# This file shows the expected structure of your dataset

import pandas as pd
import numpy as np

def create_sample_dataset():
    """
    Creates a sample dataset with the expected structure.
    Use this as a reference for your actual dataset format.
    """
    
    # Sample data - replace with your actual data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(20, 80, n_samples),
        'History': np.random.choice(['Yes', 'No'], n_samples),
        'Patient': np.random.randint(0, 5, n_samples),
        'TakeMedication': np.random.choice(['Yes', 'No'], n_samples),
        'Severity': np.random.randint(0, 4, n_samples),
        'BreathShortness': np.random.choice(['Yes', 'No'], n_samples),
        'VisualChanges': np.random.choice(['Yes', 'No'], n_samples),
        'NoseBleeding': np.random.choice(['Yes', 'No'], n_samples),
        'Whendiagnoused': np.random.randint(0, 10, n_samples),
        'Systolic': np.random.randint(90, 200, n_samples),
        'Diastolic': np.random.randint(60, 120, n_samples),
        'ControlledDiet': np.random.choice(['Yes', 'No'], n_samples),
        'Stages': np.random.choice([
            'NORMAL', 
            'HYPERTENSION (Stage-1)', 
            'HYPERTENSION (Stage-2)', 
            'HYPERTENSIVE CRISIS'
        ], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    return df

def save_sample_dataset():
    """Save a sample dataset to CSV file"""
    df = create_sample_dataset()
    df.to_csv('sample_blood_pressure_data.csv', index=False)
    print("Sample dataset saved as 'sample_blood_pressure_data.csv'")
    print("\nDataset structure:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    save_sample_dataset()