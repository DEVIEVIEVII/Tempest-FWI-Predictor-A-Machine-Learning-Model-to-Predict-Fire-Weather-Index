from flask import Flask, render_template, request
import numpy as np
import pickle

# =========================
# Load model and scaler
# =========================
try:
    model = pickle.load(open("ridge.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    print("Model and scaler loaded successfully")
except Exception as e:
    print("Error loading model:", e)

# =========================
# Create Flask App
# =========================
app = Flask(__name__)

# =========================
# Home Page
# =========================
@app.route('/')
def home():
    return render_template("home.html")

# =========================
# Input Page
# =========================
@app.route('/predict_page')
def predict_page():
    return render_template("index.html")

# =========================
# Prediction Logic
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        data = [float(x) for x in request.form.values()]
        
        print("Input Data:", data)  # Debug

        # Check feature count
        if len(data) != 9:
            return "Error: Please enter all 9 input values correctly."

        # Convert to array
        final_input = np.array(data).reshape(1, -1)

        # Scale input
        final_input = scaler.transform(final_input)

        # Predict
        prediction = model.predict(final_input)[0]

        # Risk category
        if prediction <= 2:
            risk = "Low"
        elif prediction <= 5:
            risk = "Moderate"
        elif prediction <= 10:
            risk = "High"
        else:
            risk = "Very High"

        return render_template(
            "result.html",
            prediction=round(prediction, 2),
            risk=risk
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)