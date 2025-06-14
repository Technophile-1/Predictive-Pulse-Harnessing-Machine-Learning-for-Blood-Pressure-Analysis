# Blood Pressure Analysis System

A machine learning-based web application for predicting blood pressure stages using various health parameters.

## Features

- **Data Analysis**: Comprehensive analysis of blood pressure data with visualizations
- **Machine Learning Models**: Multiple ML algorithms including Random Forest, Logistic Regression, Decision Tree, and Naive Bayes
- **Web Interface**: User-friendly Flask web application for predictions
- **Real-time Predictions**: Instant blood pressure stage predictions based on input parameters

## Project Structure

```
Blood Pressure Analysis/
├── app.py                          # Flask web application
├── blood_pressure_analysis.py      # Data analysis and model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── model.pkl                       # Trained model (generated after training)
└── templates/                      # HTML templates
    ├── index.html                  # Main prediction form
    ├── details.html                # System information page
    └── prediction.html             # Results display page
```

## Installation

1. **Clone or download the project files**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Prepare Your Dataset

1. Place your blood pressure dataset (CSV format) in the project directory
2. Update the file path in `blood_pressure_analysis.py` in the `load_and_prepare_data()` function
3. Ensure your dataset has the following columns:
   - Gender, Age, History, Patient, TakeMedication, Severity
   - BreathShortness, VisualChanges, NoseBleeding, Whendiagnoused
   - Systolic, Diastolic, ControlledDiet, Stages

### Step 2: Train the Model

Run the analysis script to train the machine learning models:

```bash
python blood_pressure_analysis.py
```

This will:
- Load and preprocess the data
- Train multiple ML models
- Save the best model as `model.pkl`
- Display model performance metrics

### Step 3: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Web Application Usage

1. **Home Page**: Fill in the health parameters form
2. **Details Page**: Learn about the system and input features
3. **Prediction Page**: View the predicted blood pressure stage

### Input Parameters

- **Gender**: Male (1) or Female (0)
- **Age**: Patient's age in years
- **History**: Family history of hypertension (Yes=1, No=0)
- **Patient Type**: Classification number
- **Take Medication**: Currently on medication (Yes=1, No=0)
- **Severity**: Symptom severity level
- **Breath Shortness**: Shortness of breath (Yes=1, No=0)
- **Visual Changes**: Visual disturbances (Yes=1, No=0)
- **Nose Bleeding**: Nosebleeds occurrence (Yes=1, No=0)
- **When Diagnosed**: Time since diagnosis
- **Systolic Pressure**: Systolic BP reading
- **Diastolic Pressure**: Diastolic BP reading
- **Controlled Diet**: Following controlled diet (Yes=1, No=0)

### Prediction Categories

- **NORMAL**: Blood pressure within normal range
- **HYPERTENSION (Stage-1)**: Mild hypertension
- **HYPERTENSION (Stage-2)**: Moderate to severe hypertension
- **HYPERTENSIVE CRISIS**: Severe hypertension requiring immediate attention

## Machine Learning Models

The system uses multiple algorithms and selects the best performing one:

1. **Random Forest Classifier** (Primary model)
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **Gaussian Naive Bayes**
5. **Multinomial Naive Bayes**

## Important Notes

- **Medical Disclaimer**: This system is for educational purposes only and should not replace professional medical advice
- **Data Privacy**: Ensure patient data is handled according to healthcare privacy regulations
- **Model Accuracy**: Regularly retrain models with new data for better accuracy

## Troubleshooting

### Common Issues

1. **Model not found error**: Run the training script first to generate `model.pkl`
2. **Import errors**: Install all dependencies using `pip install -r requirements.txt`
3. **Dataset loading issues**: Check file path and format in the analysis script

### Support

For issues or questions, please check:
1. Ensure all dependencies are installed
2. Verify dataset format and file paths
3. Check console output for error messages

## Future Enhancements

- Add more sophisticated feature engineering
- Implement cross-validation for better model evaluation
- Add data visualization dashboard
- Include more health parameters
- Implement user authentication and data storage