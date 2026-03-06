# Mental Health Prediction App

A machine learning-powered web application designed to predict the likelihood of mental health issues based on user-inputted lifestyle and demographic factors. Built with Streamlit, Scikit-learn, and other Python libraries.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Overview

This project aims to provide an accessible tool for initial mental health screening. It uses a trained machine learning model to analyze various factors such as stress levels, sleep patterns, physical activity, and occupation to predict a potential mental health condition.

**Important Disclaimer:** This application is not a substitute for professional medical advice, diagnosis, or treatment. The predictions are based on a model trained on a specific dataset and should be considered as an informational tool only. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

## Features

- **User-Friendly Interface:** Simple and intuitive web interface built with Streamlit.
- **Mental Health Prediction:** Input your data and get an instant prediction regarding your mental health status.
- **Data-Driven Insights:** The model is trained on a relevant dataset to identify patterns associated with mental health issues.
- **Easy to Deploy:** The application can be easily run locally or deployed to the cloud (e.g., Streamlit Community Cloud, Heroku).

## Live Demo

A live version of the app is hosted on Streamlit Community Cloud:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app/)


## Installation & Local Setup

Follow these steps to run the application on your local machine.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WillKimeto/Mental-Health-Prediction-App.git
   cd Mental-Health-Prediction-App
2. **Create a virtual environment**
   python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. **Install the required dependencies:**
pip install -r requirements.txt
4. **Run the Streamlit application:**
streamlit run app.py
5. **Open your browser**

   ### Project Structure
   Mental-Health-Prediction-App/
├── app.py                 # Main Streamlit application script
├── mental_health_model.pkl # Trained machine learning model (pickle file)
├── train_model.py         # Script used to train and save the model
├── requirements.txt       # Python dependencies list
├── README.md             # Project documentation (this file)
└── data/                 # Directory containing the dataset (if included)
    └── mental_health_data.csv

   ### Model Training
   The machine learning model was trained using the train_model.py script. This script handles:

Loading and preprocessing the dataset.

Training a classification model (e.g., Random Forest, Logistic Regression).

Evaluating the model's performance.

Saving the trained model as a .pkl file for use in the web app.

To retrain the model with different parameters or a new dataset, simply run:
python train_model.py

### Usage
Launch the application (see Installation & Local Setup).

Fill in the form on the web page with the required information, such as:

Age

Gender

Occupation

Sleep Duration (hours)

Physical Activity Level

Stress Level (on a scale of 1-10)

etc.

Click the "Predict" button.

The application will display the prediction (e.g., "Low Risk," "Potential Issue Detected").

### Dataset
The model is trained on a dataset containing anonymized information related to mental health factors. Due to privacy concerns, the original dataset may not be included in this repository. If you wish to use your own data, ensure it is formatted correctly and update the train_model.py script accordingly.

### Author
Will Kimeto

GitHub: @WillKimeto

### Disclaimer
This tool is for informational and educational purposes only. It is not a certified medical device. The predictions made by this application should not be considered a medical diagnosis. If you are experiencing a mental health crisis, please contact a licensed healthcare professional or a crisis service immediately.



