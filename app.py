import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
# Load the trained model
model_filename = 'random_forest_model.joblib'
loaded_model = joblib.load(model_filename)

# Load the data preprocessing steps
label_encoder, encoded_columns = joblib.load('data_preprocessing.joblib')
label_encoder = LabelEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Title
st.title("Shark Tank India Prediction App")

# Sidebar
st.sidebar.header("User Input")

# Input form
startup_name = st.sidebar.text_input("Startup Name", "Your Startup Name")
number_of_presenters = st.sidebar.number_input("Number of Presenters", min_value=0, max_value=10, value=1)
pitchers_average_age = st.sidebar.text_input("Pitchers Average Age", "25-30")
industry = st.sidebar.selectbox("Industry", ['Agriculture', 'Animal/Pets', 'Beauty/Fashion', 'Education', 'Electronics', 'Entertainment', 'Food', 'Furnishing/Household', 'Hardware', 'Liquor/Beverages', 'Manufacturing', 'Medical/Health', 'Services', 'Sports', 'Technology/Software', 'Vehicles/Electrical Vehicles'])
region = st.sidebar.selectbox("Region", ['Central', 'East', 'North', 'Northeast', 'South', 'West'])
deal_has_conditions = st.sidebar.selectbox("Deal has conditions", ['no', 'yes'])

# Prepare data for prediction
new_data = pd.DataFrame({
    'Startup Name': [startup_name],
    'Number of Presenters': [number_of_presenters],
    'Pitchers Average Age': [pitchers_average_age],
    'Industry': [industry],
    'Region': [region],
    'Deal has conditions': [deal_has_conditions]
})

# Apply the label encoding to the 'Pitchers Average Age' column
#new_data['Pitchers Average Age'] = label_encoder.transform(new_data['Pitchers Average Age'])

new_data['Pitchers Average Age'] = new_data['Pitchers Average Age'].apply(lambda x: x if x in label_encoder.classes_ else 'default_value')
new_data['Pitchers Average Age'] = label_encoder.transform(new_data['Pitchers Average Age'])

# Perform one-hot encoding on new data
new_data = pd.get_dummies(new_data, columns=['Industry', 'Region', 'Deal has conditions'])

# Ensure the columns match the encoded columns from the preprocessing steps
missing_columns = set(encoded_columns) - set(new_data.columns)
for column in missing_columns:
    new_data[column] = 0  # Add missing columns with all zeros

if st.button("Make Prediction"):
    prediction = loaded_model.predict(new_data)
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("Accepted Offer")
    else:
        st.write("Not Accepted Offer")

# Data source and information
st.markdown("Data source: Your data source here")
st.markdown("This is a Streamlit app for predicting the outcome of a pitch on Shark Tank India using a trained machine learning model.")


if __name__ == '__main__':
    main()
