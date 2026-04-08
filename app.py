from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

ridge_model = pickle.load(open('ridge.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['day']),
            float(request.form['month']),
            float(request.form['year']),
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['DC']),
            float(request.form['ISI']),
            float(request.form['BUI']),
            float(request.form['Classes']),
            float(request.form['Region']),
        ]

        data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(data)
        result = ridge_model.predict(scaled_data)[0]
        fwi = round(float(result), 2)

        if fwi < 5.4:
            risk_level = "Low"
            risk_color = "#4ade80"
            risk_desc = "Minimal fire danger. Conditions are generally safe."
        elif fwi < 11.2:
            risk_level = "Moderate"
            risk_color = "#facc15"
            risk_desc = "Some fire danger present. Stay alert to conditions."
        elif fwi < 21.3:
            risk_level = "High"
            risk_color = "#fb923c"
            risk_desc = "High fire danger. Avoid outdoor burning activities."
        elif fwi < 38.0:
            risk_level = "Very High"
            risk_color = "#f97316"
            risk_desc = "Very high fire danger. Fire can spread rapidly."
        else:
            risk_level = "Extreme"
            risk_color = "#ef4444"
            risk_desc = "Extreme fire danger. Conditions are critical."

        input_data = {
            'temperature': request.form['Temperature'],
            'humidity': request.form['RH'],
            'wind': request.form['Ws'],
            'rain': request.form['Rain'],
            'ffmc': request.form['FFMC'],
            'dmc': request.form['DMC'],
            'dc': request.form['DC'],
            'isi': request.form['ISI'],
            'bui': request.form['BUI'],
        }

        return render_template('home.html',
                               result=fwi,
                               risk_level=risk_level,
                               risk_color=risk_color,
                               risk_desc=risk_desc,
                               input_data=input_data)

    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)