import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'random_forest_model.joblib'
loaded_model = joblib.load(model_filename)
predcol = ['Number of Presenters', 'Male Presenters', 'Female Presenters',
       'Transgender Presenters', 'Couple Presenters', 'Pitchers Average Age',
       'Started in', 'Yearly Revenue', 'Monthly Sales', 'Gross Margin',
       'Net Margin', 'Original Ask Amount', 'Original Offered Equity',
       'Valuation Requested', 'Has Patents', 'Industry_Agriculture',
       'Industry_Animal/Pets', 'Industry_Beauty/Fashion', 'Industry_Education',
       'Industry_Electronics', 'Industry_Entertainment', 'Industry_Food',
       'Industry_Furnishing/Household', 'Industry_Hardware',
       'Industry_Liquor/Beverages', 'Industry_Manufacturing',
       'Industry_Medical/Health', 'Industry_Services', 'Industry_Sports',
       'Industry_Technology/Software', 'Industry_Vehicles/Electrical Vehicles',
       'Region_Central', 'Region_East', 'Region_North', 'Region_Northeast',
       'Region_South', 'Region_West', 'Deal has conditions_no',
       'Deal has conditions_yes']

# Load the data preprocessing steps
_, encoded_columns = joblib.load('data_preprocessing.joblib')

# Define a custom label encoding function for 'Pitchers Average Age'
def custom_label_encode(value):
    if value == '25-30':
        return 0
    elif value == '30-35':
        return 1
    elif value == '35-40':
        return 2
    # Add more conditions as needed
    else:
        # Handle unknown values
        return -1

# Title
st.title("Shark Tank India Prediction App")

# Sidebar
st.sidebar.header("User Input")

# Input form
#startup_name = st.sidebar.text_input("Startup Name", "Your Startup Name")
number_of_presenters = st.sidebar.number_input("Number of Presenters", min_value=0, max_value=10, value=1)
pitchers_average_age = st.sidebar.selectbox("Pitchers Average Age", ['25-30', '30-35', '35-40', 'Other'])
started_in = st.sidebar.number_input("Started in", min_value=0, value=2020)
yearly_revenue = st.sidebar.number_input("Yearly Revenue", min_value=0, value=100)
monthly_sales = st.sidebar.text_input("Monthly Sales", "Monthly sales in lakhs")
gross_margin = st.sidebar.number_input("Gross Margin", min_value=0, value=10)
net_margin = st.sidebar.number_input("Net Margin", min_value=0, value=5)
original_ask_amount = st.sidebar.number_input("Original Ask Amount", min_value=0, value=100)
original_offered_equity = st.sidebar.number_input("Original Offered Equity", min_value=0, value=20)
valuation_requested = st.sidebar.number_input("Valuation Requested", min_value=0, value=200)
# total_deal_amount = st.sidebar.number_input("Total Deal Amount", min_value=0, value=300)
# total_deal_equity = st.sidebar.number_input("Total Deal Equity", min_value=0, value=50)
# total_deal_debt = st.sidebar.number_input("Total Deal Debt", min_value=0, value=30)
# debt_interest = st.sidebar.number_input("Debt Interest", min_value=0, value=2)
# deal_valuation = st.sidebar.number_input("Deal Valuation", min_value=0, value=250)
# number_of_sharks_in_deal = st.sidebar.number_input("Number of Sharks in Deal", min_value=0, value=3)
has_patents = st.sidebar.selectbox("Has Patents", ['No', 'Yes'])
industry = st.sidebar.selectbox("Industry", ['Agriculture', 'Animal/Pets', 'Beauty/Fashion', 'Education', 'Electronics', 'Entertainment', 'Food', 'Furnishing/Household', 'Hardware', 'Liquor/Beverages', 'Manufacturing', 'Medical/Health', 'Services', 'Sports', 'Technology/Software', 'Vehicles/Electrical Vehicles'])
region = st.sidebar.selectbox("Region", ['Central', 'East', 'North', 'Northeast', 'South', 'West'])
deal_has_conditions = st.sidebar.selectbox("Deal has conditions", ['no', 'yes'])

# Prepare data for prediction
new_data = pd.DataFrame({
    #'Startup Name': [startup_name],
    'Number of Presenters': [number_of_presenters],
    'Pitchers Average Age': [pitchers_average_age],
    'Started in': [started_in],
    'Yearly Revenue': [yearly_revenue],
    'Gross Margin': [gross_margin],
    'Net Margin': [net_margin],
    'Original Ask Amount': [original_ask_amount],
    'Original Offered Equity': [original_offered_equity],
    'Valuation Requested': [valuation_requested],
    # 'Total Deal Amount': [total_deal_amount],
    # 'Total Deal Equity': [total_deal_equity],
    # 'Total Deal Debt': [total_deal_debt],
    # 'Debt Interest': [debt_interest],
    # 'Deal Valuation': [deal_valuation],
    # 'Number of Sharks in Deal': [number_of_sharks_in_deal],
    'Has Patents': [has_patents],
    'Industry': [industry],
    'Region': [region],
    'Deal has conditions': [deal_has_conditions]
})

# Apply custom label encoding to the 'Pitchers Average Age' column
new_data['Pitchers Average Age'] = new_data['Pitchers Average Age'].apply(custom_label_encode)

# Perform one-hot encoding on new data
new_data = pd.get_dummies(new_data, columns=['Industry', 'Region', 'Deal has conditions', 'Has Patents'])

# Ensure the columns match the encoded columns from the preprocessing steps
missing_columns = set(encoded_columns) - set(new_data.columns)
for column in missing_columns:
    new_data[column] = 0  # Add missing columns with all zeros

if st.button("Make Prediction"):
    prediction = loaded_model.predict(new_data[predcol])
    confidence = loaded_model.predict_proba(new_data[predcol])
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write(f"Accepted Offer✔ with confidence score {round(confidence, 4)*100}%")
    else:
        st.write("Not Accepted Offer❌")

# Data source and information
# st.markdown("Data source: Your data source here")
markdown_string = """## Sharktank Offer Outcome Prediction Model

This is a Streamlit app for predicting the outcome of a pitch on Shark Tank India using a trained machine learning model.

Find the code [here](https://github.com/Pranav-V-27/CRISP-DM)
"""
st.markdown(markdown_string)

#if __name__ == '__main__':
    #main()
