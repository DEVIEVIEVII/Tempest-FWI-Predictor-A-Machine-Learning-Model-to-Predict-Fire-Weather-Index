from flask import Flask, render_template, request

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction page
@app.route('/predict', methods=['POST'])
def predict():

    def get_val(name):
        val = request.form.get(name)
        return float(val) if val else 0

    # 15 inputs
    data = {
        "day": get_val('day'),
        "month": get_val('month'),
        "year": get_val('year'),
        "temp": get_val('temp'),
        "humidity": get_val('humidity'),
        "wind": get_val('wind'),
        "rain": get_val('rain'),
        "ffmc": get_val('ffmc'),
        "dmc": get_val('dmc'),
        "isi": get_val('isi'),
        "dc": get_val('dc'),
        "bui": get_val('bui'),
        "fwi": get_val('fwi'),
        "region": get_val('region'),
        "classes": get_val('classes')
    }

    # Simple logic
    if data["temp"] > 30:
        result = " HIGH FIRE RISK"
        burnt_area = round(data["temp"] * 0.8, 2)
    else:
        result = " LOW FIRE RISK"
        burnt_area = round(data["temp"] * 0.3, 2)

    return render_template('result.html',
                           result=result,
                           burnt_area=burnt_area,
                           data=data)


if __name__ == '__main__':
    app.run(debug=True)