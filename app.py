from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/ridge.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])

        # Arrange input in SAME ORDER as training (Temperature, RH, Ws, Rain, FFMC, DMC, ISI)
        data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI]])

        # Scale input
        scaled_data = scaler.transform(data)

        # Predict
        prediction = model.predict(scaled_data)[0]

        return render_template("home.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)