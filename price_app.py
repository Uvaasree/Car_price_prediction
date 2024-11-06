import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

# Streamlit app setup
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")
st.title(" Car Price Prediction App")
st.markdown("### Predict the price of a car based on its features!")


def scaling(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    list_of_columns = ['Engine Displacement', 'Mileage', 'Max Power', 'Wheel Base', 'Kerb Weight', 'Cargo Volumn']
    df[list_of_columns] = scaler.fit_transform(df[list_of_columns])
    return df

def encoding(df):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    columns_to_encode = [
        'bt', 'transmission','Gear Box', 'oem', 'model', 'Insurance Validity', 'Fuel Type',
        'Seats', 'RTO', 'features', 'Comfort & Convenience', 'Exterior', 'Safety',
        'No Door Numbers', 'modelyear', 'Registration year'
    ]
    for col in columns_to_encode:
        df[col] = label_encoder.fit_transform(df[col])
    return df

# Load the dataset and model
df = pd.read_excel('review_price.xlsx')
df = df.drop(columns=['Unnamed: 0','price'], axis=1)

# Input widgets
engine_displacement = st.slider("Engine Displacement", min_value=min(df['Engine Displacement']), max_value=max(df['Engine Displacement']), step=10, value=1199)
comfort_convenience = st.slider("Comfort & Convenience features", min(df['Comfort & Convenience']), max(df['Comfort & Convenience']), step=1, value=13)
safety = st.slider("Safety features", min(df['Safety']), max(df['Safety']), step=1, value=26)
max_power = st.slider("Max Power", min_value=min(df['Max Power']), max_value=max(df['Max Power']), step=10.0, value=47.33)

# Corrected selectbox fields with index calculations
rto_options = list(df['RTO'].unique())
rto = st.selectbox("RTO", options=rto_options, index=rto_options.index("TS09"))  # Replace "TS09" with correct default

registration_year_options = list(df['Registration year'].unique())
registration_year = st.selectbox("Registration Year", options=registration_year_options, index=registration_year_options.index(2016))

no_door_numbers_options = list(df['No Door Numbers'].unique())
no_door_numbers = st.selectbox("Number of Doors", options=no_door_numbers_options, index=no_door_numbers_options.index(5))

fuel_type_options = list(df['Fuel Type'].unique())
fuel_type = st.selectbox("Fuel Type", options=fuel_type_options, index=fuel_type_options.index("Petrol"))

insurance_validity_options = list(df['Insurance Validity'].unique())
insurance_validity = st.selectbox("Insurance Validity", options=insurance_validity_options, index=insurance_validity_options.index("Third Party insurance"))

features_options = list(df['features'].unique())
features = st.selectbox("Additional Features", options=features_options, index=features_options.index(9))

mileage = st.slider("Mileage (km/l)", min_value=min(df['Mileage']), max_value=max(df['Mileage']), step=1.0, value=21.1)
exterior = st.slider("Exterior features", min(df['Exterior']), max(df['Exterior']), step=1, value=16)

oem_options = list(df['oem'].unique())
oem = st.selectbox("OEM Brand", options=oem_options, index=oem_options.index("Renault"))

transmission_options = list(df['transmission'].unique())
transmission = st.selectbox("Transmission", options=transmission_options, index=transmission_options.index("Automatic"))

bt_options = list(df['bt'].unique())
bt = st.selectbox("Body Type", options=bt_options, index=bt_options.index("Hatchback"))

model_options = list(df['model'].unique())
model = st.selectbox("Model", options=model_options, index=model_options.index("Maruti Brezza"))

modelyear_options = list(df['modelyear'].unique())
modelyear = st.selectbox("Model Year", options=modelyear_options, index=modelyear_options.index(1970))

seats_options = list(df['Seats'].unique())
seats = st.selectbox("Seats", options=seats_options, index=seats_options.index(5))

gear_box_options = list(df['Gear Box'].unique())
gear_box = st.selectbox("Gear Box (Speeds)", options=gear_box_options, index=gear_box_options.index("5 Speed"))

cargo_volumn = st.slider("Cargo Volume (L)", min_value=min(df['Cargo Volumn']), max_value=max(df['Cargo Volumn']), step=10.0, value=780.0)
wheel_base = st.slider("Wheel Base (mm)", min_value=min(df['Wheel Base']), max_value=max(df['Wheel Base']), step=10, value=2470)
kerb_weight = st.slider("Kerb Weight (kg)", min_value=min(df['Kerb Weight']), max_value=max(df['Kerb Weight']), step=10, value=1190)

# Organize user input for prediction
user_input = pd.DataFrame([[engine_displacement, comfort_convenience, safety, max_power, rto,
                            registration_year, no_door_numbers, fuel_type, insurance_validity, features,
                            mileage, exterior, oem, transmission, bt, model, modelyear, seats,
                            gear_box, cargo_volumn, wheel_base, kerb_weight]], 
                          columns=[
                            'Engine Displacement', 'Comfort & Convenience', 'Safety', 'Max Power', 'RTO',
                            'Registration year', 'No Door Numbers', 'Fuel Type', 'Insurance Validity', 'features',
                            'Mileage', 'Exterior', 'oem', 'transmission', 'bt', 'model', 'modelyear', 'Seats',
                            'Gear Box', 'Cargo Volumn', 'Wheel Base', 'Kerb Weight'
                          ])

# Scale and encode input
user_input = scaling(user_input)
user_input = encoding(user_input)

with open("best_model_1.pkl", 'rb') as file:
    model = pickle.load(file)

# Predict
if st.button("Predict Price"):
    try:
        predicted_price = model.predict(user_input)
        st.subheader(f"Estimated Car Price: Rupees {predicted_price[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
