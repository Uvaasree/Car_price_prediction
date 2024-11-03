import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer


def scaling(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    list_of_columns = ['Engine Displacement', 'Mileage', 'Max Power', 'Wheel Base', 'Kerb Weight', 'Cargo Volumn']


    df[list_of_columns]=scaler.fit_transform(df[list_of_columns])
    return df

def encoding(df):
    from sklearn.preprocessing import LabelEncoder
    # Label Encoding
    label_encoder = LabelEncoder()

    df['bt']=label_encoder.fit_transform(df['bt'])
    df['transmission']=label_encoder.fit_transform(df['transmission'])
    df['oem']=label_encoder.fit_transform(df['oem'])
    df['model']=label_encoder.fit_transform(df['model'])
    df['Insurance Validity']=label_encoder.fit_transform(df['Insurance Validity'])
    df['Fuel Type']=label_encoder.fit_transform(df['Fuel Type'])
    df['Seats']=label_encoder.fit_transform(df['Seats'])
    df['RTO']=label_encoder.fit_transform(df['RTO'])
    df['features']=label_encoder.fit_transform(df['features'])
    df['Comfort & Convenience']=label_encoder.fit_transform(df['Comfort & Convenience']) 
    df['Exterior']=label_encoder.fit_transform(df['Exterior'])
    df['Safety']=label_encoder.fit_transform(df['Safety'])
    df['No Door Numbers']=label_encoder.fit_transform(df['No Door Numbers'])
    df['modelyear']=label_encoder.fit_transform(df['modelyear'])
    df['Registration year']=label_encoder.fit_transform(df['Registration year'])
    
    return df


df=pd.read_excel('E:/car_price_prediction/cleaned_data.xlsx')

# Load the trained model
try:
    with open('best_gb_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path and try again.")

# Streamlit app setup
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# Title and Description
st.title("🚗 Car Price Prediction App")
st.markdown("### Predict the price of a car based on its features!")

# Define default values and options for each feature based on data types
default_values = {
    'Engine Displacement': np.random.choice(df['Engine Displacement']),
    'Comfort & Convenience': np.random.choice(df['Comfort & Convenience']),
    'Safety': np.random.choice(df['Safety']),
    'Max Power': np.random.choice(df['Max Power']),
    'RTO': np.random.choice(df['RTO']),
    'Registration year': np.random.choice(df['Registration year']),
    'No Door Numbers': np.random.choice(df['No Door Numbers']),
    'Fuel Type': np.random.choice(df['Fuel Type']),
    'Insurance Validity': np.random.choice(df['Insurance Validity']),
    'features': np.random.choice(df['features']),
    'Mileage': np.random.choice(df['Mileage']),
    'Exterior': np.random.choice(df['Exterior']),
    'oem': np.random.choice(df['oem']),
    'transmission': np.random.choice(df['transmission']),
    'bt': np.random.choice(df['bt']),
    'model': np.random.choice(df['model']),
    'modelyear': np.random.choice(df['modelyear']),
    'Seats': np.random.choice(df['Seats']),
    'Gear Box': np.random.choice(df['Gear Box']),
    'Cargo Volumn': np.random.choice(df['Cargo Volumn']),
    'Wheel Base': np.random.choice(df['Wheel Base']),
    'Kerb Weight': np.random.choice(df['Kerb Weight'])
}

# Input fields for each feature with appropriate widgets
engine_displacement = st.slider("Engine Displacement", min_value=min(df['Engine Displacement']), max_value=max(df['Engine Displacement']), step=10, value=default_values['Engine Displacement'])

comfort_convenience = st.slider("Comfort & Convenience features", min(df['Comfort & Convenience']), max(df['Comfort & Convenience']), default_values['Comfort & Convenience'])

safety = st.slider("Safety features", min(df['Safety']), max(df['Safety']), default_values['Safety'])

max_power = st.slider("Max Power", min_value=min(df['Max Power']), max_value=min(df['Max Power']), step=10, value=default_values['Max Power'])

rto = st.selectbox("RTO ", options=list(df['RTO'].unique()), index=default_values['RTO'])

registration_year = st.selectbox("Registration Year",options=list(df['Registration year'].unique()), value=default_values['Registration year'])

no_door_numbers = st.selectbox("Number of Doors", options=list(df['No Door Numbers'].unique()), index=default_values['No Door Numbers'])

fuel_type = st.selectbox("Fuel Type", options=list(df['Fuel Type'].unique()), index=default_values['Fuel Type'])

insurance_validity = st.selectbox("Insurance Validity",options=list(df['Insurance Validity'].unique()), index=default_values['Insurance Validity'])

features = st.selectbox("Additional Features ",options=list(df['features'].unique()), index=default_values['features'])

mileage = st.slider("Mileage (km/l)", min_value=min(df['Mileage']), max_value=max(df['Mileage']), step=1, value=default_values['Mileage'])

exterior = st.slider("Exterior features", min(df['Exterior']), max(df['Exterior']), default_values['Exterior'])

oem = st.selectbox("OEM Brand ", options=list(df['oem'].unique()), index=default_values['oem'])

transmission = st.selectbox("Transmission", options=list(df['transmission'].unique()), index=default_values['transmission'])

bt = st.selectbox("Body Type", options=list(df['bt'].unique()), index=default_values['bt'])

model = st.selectbox("Model", options=list(df['model']), index=default_values['model'])

modelyear = st.selectbox("Model Year", options=list(df['modelyear'].unique()), index=default_values['modelyear'])

seats = st.selectbox("Seats", options=list(df['Seats'].unique()), index=default_values['Seats'])

gear_box = st.selectbox("Gear Box (Speeds)", options=list(df['Gear Box'].unique()), index=default_values['Gear Box'])

cargo_volumn = st.slider("Cargo Volume (L)", min_value=min(df['Cargo Volumn']), max_value=max(df['Cargo Volumn']), step=10, value=default_values['Cargo Volumn'])

wheel_base = st.slider("Wheel Base (mm)", min_value=min(df['Wheel Base']), max_value=max(df['Wheel Base']), step=10, value=default_values['Wheel Base'])

kerb_weight = st.slider("Kerb Weight (kg)", min_value=min(df['Kerb Weight']), max_value=max(df['Kerb Weight']), step=10, value=default_values['Kerb Weight'])


# Organize user input into a feature array for prediction
user_input = pd.DataFrame([
    engine_displacement, comfort_convenience, safety, max_power, rto,
    registration_year, no_door_numbers, fuel_type, insurance_validity, features,
    mileage, exterior, oem, transmission, bt, model, modelyear, seats,
    gear_box, cargo_volumn, wheel_base, kerb_weight
])

#scaling and encoding user input
user_input = scaling(user_input)
user_input = encoding(user_input)

# Predict the price based on user input
if st.button("Predict Price"):
    try:
        predicted_price = model.predict(user_input)
        st.subheader(f"Estimated Car Price: Rupees {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
