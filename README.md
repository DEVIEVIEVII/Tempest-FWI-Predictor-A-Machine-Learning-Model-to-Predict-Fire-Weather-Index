# Tempest-FWI-Predictor-A-Machine-Learning-Model-to-Predict-Fire-Weather-Index
#  Fire Weather Index (FWI) Prediction System

## Overview

The **Fire Weather Index (FWI) Prediction System** is a machine learning-based web application that predicts wildfire risk and estimated burnt area using environmental parameters such as temperature, humidity, wind speed, and rainfall.

The system is deployed as a Flask web application, enabling real-time predictions through an interactive user interface.

##  Live Application

🔗 **Deployed Link:**
https://fwi-predictor-m4ck.onrender.com

> *(Note: The application may take a few seconds to load initially due to free hosting instance limitations.)*

##  Objectives

* Predict wildfire risk levels (Low / High)
* Estimate potential burnt area
* Provide a user-friendly dashboard for analysis
* Enable real-time prediction using ML models

## Technologies Used

 *Programming Language:** Python
*Framework:** Flask
*Machine Learning:** Scikit-learn
*Frontend:** HTML, CSS
*Deployment:** Render
*Version Control:** Git & GitHub

##  Key Features

*  Real-time fire risk prediction
*  Burnt area estimation
*  Interactive and clean dashboard UI
*  Risk analysis with confidence level
*  Deployment on cloud platform

##  System Workflow

1. User enters environmental parameters
2. Flask backend receives input data
3. Data is preprocessed using a scaler
4. Trained ML model generates predictions
5. Results are displayed dynamically on the dashboard

##  Project Structure
templates/
 ├── index.html
 ├── home.html
 ├── result.html
app.py
model.pkl
scaler.pkl
requirements.txt
Procfile

## Limitations

* Initial load time may be slow due to free hosting
* Predictions depend on dataset quality and training

## Future Enhancements

* Integration of real-time weather APIs
* Advanced visualization using charts and graphs
* Mobile-friendly responsive UI
* Improved prediction accuracy using advanced models
## Conclusion

This project demonstrates the practical application of machine learning in predicting wildfire risk. By combining data-driven insights with a web interface, it provides a useful tool for early warning and decision support in fire management systems.

