import os
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, static_url_path='/static')

# Load the trained model and label encoders
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model file not found or error: {e}")
    model = None

try:
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    print("Label encoders loaded successfully!")
except Exception as e:
    print(f"Label encoders file not found or error: {e}")
    label_encoders = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = {
            "Gender": request.form["Gender"],
            "Age": request.form["Age"],
            "History": request.form["History"],
            "Patient": request.form["Patient"],
            "TakeMedication": request.form["TakeMedication"],
            "Severity": request.form["Severity"],
            "BreathShortness": request.form["BreathShortness"],
            "VisualChanges": request.form["VisualChanges"],
            "NoseBleeding": request.form["NoseBleeding"],
            "Whendiagnoused": request.form["Whendiagnoused"],
            "Systolic": request.form["Systolic"],
            "Diastolic": request.form["Diastolic"],
            "ControlledDiet": request.form["ControlledDiet"]
        }
        print("Received form data:", form_data)

        df = pd.DataFrame([form_data])
        for col in df.columns:
            df[col] = df[col].astype(str)

        if label_encoders:
            for col in label_encoders:
                if col in df.columns:
                    df[col] = label_encoders[col].transform(df[col])

        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                pass

        print("Input features for prediction:")
        print(df)

        if model is not None:
            prediction = model.predict(df)
            print(f"Prediction: {prediction[0]}")

            if prediction[0] == 0:
                result = "NORMAL"
            elif prediction[0] == 1:
                result = "HYPERTENSION (Stage-1)"
            elif prediction[0] == 2:
                result = "HYPERTENSION (Stage-2)"
            else:
                result = "HYPERTENSIVE CRISIS"

            text = "Your Blood Pressure stage is: "
            return render_template("prediction.html", prediction_text=text + result)
        else:
            return render_template("prediction.html",
                                   prediction_text="Model not available. Please train the model first.")

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return render_template("prediction.html",
                               prediction_text=f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)