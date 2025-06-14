# Blood Pressure Analysis - Data Processing and Model Training
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare the dataset"""
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data"""
    if df is None:
        return None, None
    
    # Rename column 'C' as Gender
    if 'C' in df.columns:
        df.rename(columns={'C': 'Gender'}, inplace=True)
    
    # Check data info
    print("Dataset Info:")
    print(df.info())
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nNull Values:\n{df.isnull().sum()}")
    
    # Display unique values for each column to understand the data
    print("\nUnique values in each column:")
    for col in df.columns:
        print(f"{col}: {df[col].unique()[:10]}")  # Show first 10 unique values
    
    # Clean the Stages column
    if 'Stages' in df.columns:
        df['Stages'].replace({
            'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS', 
            'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'
        }, inplace=True)
        print(f"\nUnique Stages after cleaning: {df['Stages'].unique()}")
    
    # Convert categorical columns to numerical using LabelEncoder
    categorical_columns = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 
                          'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding', 
                          'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet']
    
    # Store label encoders for each column
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
            print(f"Encoded {col}: {df[col].unique()}")
    
    # Encode target variable separately
    if 'Stages' in df.columns:
        target_encoder = LabelEncoder()
        df['Stages'] = target_encoder.fit_transform(df['Stages'])
        label_encoders['Stages'] = target_encoder
        print(f"Encoded Stages: {df['Stages'].unique()}")
    
    return df, label_encoders

def visualize_data(df):
    """Create visualizations"""
    if df is None:
        return
    
    # Gender distribution pie chart
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Gender Distribution")
        plt.axis('equal')
        plt.show()
    
    # Count plot of TakeMedication by Severity
    if 'TakeMedication' in df.columns and 'Severity' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='TakeMedication', hue='Severity', data=df)
        plt.title('Count plot of TakeMedication by Severity')
        plt.show()

def train_models(df):
    """Train multiple machine learning models"""
    if df is None:
        return None
    
    # Prepare features and target
    X = df.drop('Stages', axis=1)  # Features
    Y = df['Stages']  # Target
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
    
    models = {}
    results = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)
    acc_lr = accuracy_score(y_test, y_pred)
    c_lr = classification_report(y_test, y_pred)
    models['Logistic Regression'] = logistic_regression
    results['Logistic Regression'] = acc_lr
    print(f'Logistic Regression Accuracy Score: {acc_lr}')
    print(c_lr)
    
    # Random Forest
    print("\nTraining Random Forest...")
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)
    acc_rf = accuracy_score(y_test, y_pred)
    c_rf = classification_report(y_test, y_pred)
    models['Random Forest'] = random_forest
    results['Random Forest'] = acc_rf
    print(f'Random Forest Accuracy Score: {acc_rf}')
    print(c_rf)
    
    # Decision Tree
    print("\nTraining Decision Tree...")
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(x_train, y_train)
    y_pred = decision_tree_model.predict(x_test)
    acc_dt = accuracy_score(y_test, y_pred)
    c_dt = classification_report(y_test, y_pred)
    models['Decision Tree'] = decision_tree_model
    results['Decision Tree'] = acc_dt
    print(f'Decision Tree Accuracy Score: {acc_dt}')
    print(c_dt)
    
    # Gaussian Naive Bayes
    print("\nTraining Gaussian Naive Bayes...")
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    acc_nb = accuracy_score(y_test, y_pred)
    c_nb = classification_report(y_test, y_pred)
    models['Gaussian NB'] = nb
    results['Gaussian NB'] = acc_nb
    print(f'Gaussian NB Accuracy Score: {acc_nb}')
    print(c_nb)
    
    # Multinomial Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_pred = mnb.predict(x_test)
    acc_mnb = accuracy_score(y_test, y_pred)
    c_mnb = classification_report(y_test, y_pred)
    models['Multinomial NB'] = mnb
    results['Multinomial NB'] = acc_mnb
    print(f'Multinomial NB Accuracy Score: {acc_mnb}')
    print(c_mnb)
    
    # Compare models
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Score': list(results.values())
    })
    print("\nModel Comparison:")
    print(model_comparison)
    
    # Save the best model (Random Forest in this case)
    best_model = random_forest
    pickle.dump(best_model, open("model.pkl", "wb"))
    print("\nBest model saved as 'model.pkl'")
    
    return models, results

def main():
    """Main function to run the analysis"""
    print("Blood Pressure Analysis - Data Processing and Model Training")
    print("=" * 60)
    
    # Load and prepare data using your dataset
    df = load_and_prepare_data("patient_data.csv")
    
    if df is not None:
        # Preprocess data
        df, label_encoders = preprocess_data(df)
        
        # Visualize data
        visualize_data(df)
        
        # Train models
        models, results = train_models(df)
        
        # Save label encoders for use in the web app
        import pickle
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        print("Label encoders saved for web application use.")
        
        print("\nAnalysis completed successfully!")
    else:
        print("Please check your dataset file and try again.")

if __name__ == "__main__":
    main()