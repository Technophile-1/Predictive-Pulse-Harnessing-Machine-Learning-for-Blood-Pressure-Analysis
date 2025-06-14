# Script to analyze the dataset structure
import pandas as pd

def analyze_dataset():
    """Analyze the dataset to understand unique values in each column"""
    try:
        # Load the dataset
        df = pd.read_csv('patient_data.csv')
        
        # Rename column 'C' to 'Gender'
        if 'C' in df.columns:
            df.rename(columns={'C': 'Gender'}, inplace=True)
        
        print("Dataset Analysis")
        print("=" * 50)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\nUnique values in each column:")
        print("-" * 50)
        
        for col in df.columns:
            unique_vals = df[col].unique()
            print(f"\n{col}:")
            print(f"  Unique values: {len(unique_vals)}")
            print(f"  Values: {unique_vals}")
        
        print("\nData types:")
        print("-" * 50)
        print(df.dtypes)
        
        print("\nNull values:")
        print("-" * 50)
        print(df.isnull().sum())
        
        print("\nFirst 5 rows:")
        print("-" * 50)
        print(df.head())
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    analyze_dataset()