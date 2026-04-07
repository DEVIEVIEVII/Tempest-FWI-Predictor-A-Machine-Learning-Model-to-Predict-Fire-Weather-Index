
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle


app = Flask(__name__)

app.secret_key = "secret123"


model = pickle.load(open("ridge.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin123":
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template("login.html", message="Invalid Username or Password")

    return render_template("login.html")



@app.route('/index')
def index():

    if 'user' in session:
        return render_template("index.html")
    else:
        return redirect(url_for('login'))



@app.route('/predict', methods=['POST'])
def predict():

    if 'user' in session:
        month = int(request.form['month'])
        day = int(request.form['day'])
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        area = int(request.form['area'])
       

        data = np.array([[month,day,Temperature, RH, Ws, Rain, FFMC, DMC, ISI,area]])

        prediction = model.predict(data)

        return render_template("home.html", result=prediction[0])

    else:
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)

