# Car Price Prediction App 🚗

This repository hosts the **Car Price Prediction App**, a web application built with Streamlit that predicts the price of a car based on various features such as engine displacement, fuel type, safety features, and more. The app uses a pre-trained Gradient Boosting model to deliver accurate predictions. The model was trained and pickled prior to the development of this Streamlit app.

## Project Overview

- **Modeling**: The project began with training a Gradient Boosting Regressor on cleaned and preprocessed car data. Using RandomizedSearchCV, the model was fine-tuned for optimal performance, and then serialized (pickled) for use in the Streamlit app.
- **Streamlit App**: Built to provide an interactive interface where users can input car features and receive a predicted price in real-time.

## Features

- **Data Preprocessing**: Scaling and encoding are applied to ensure consistent and meaningful predictions.
- **User Inputs**: Various car features are provided as input fields, including sliders and dropdowns.
- **Real-time Prediction**: Predicts the car price instantly based on the user inputs.
- **Error Handling**: The app handles missing model files gracefully, alerting users if the model file is unavailable.

## Tech Stack

- **Frontend**: Streamlit for building an interactive web app.
- **Modeling**: Scikit-Learn (Gradient Boosting Regressor) with hyperparameter tuning.
- **Data Processing**: Pandas, Numpy for data manipulation; MinMaxScaler and LabelEncoder for scaling and encoding.
- **Backend**: Python for server-side logic.

## Project Structure

```
├── creating a dataframe      
├── preprocessing the data               
├── model buildng and finding best fit out of it                
├── pickling the model and building streamlit app for price prediction
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Scikit-Learn](https://scikit-learn.org/)

## Usage

Open the app in your browser, input relevant car features, and hit "Predict Price" to see the estimated price. The app handles any missing or incorrect inputs, alerting you if adjustments are needed.

---
