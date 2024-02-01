## Shark Tank India CRISP-DM Analysis

This project focuses on analyzing data and making predictive model from the data of "Shark Tank India" Seasons 1 and 2, containing information with 64 fields/columns and over 320 records. The model provides insights into the pitches made on the show, including startup details, investment offers, and outcomes.

## About Shark Tank India

"Shark Tank India" is a popular television series broadcasted on SonyLiv OTT/Sony TV. The show features entrepreneurs ("pitchers") presenting their business ideas to potential investors ("sharks") in exchange for investment offers.

## Prerequisites

Before running the project, ensure you have the required Python libraries and packages installed. You can use the following Python libraries:

- pandas
- matplotlib
- seaborn
- numpy
- babel
- wordcloud
- geopandas
- plotly
- folium

You can install additional libraries using pip as needed.

## Data Dictionary

The dataset contains 64 fields/columns, and here is an overview of each field:

- Season Number: Season number.
- Season Start: Season first aired date.
- Season End: Season last aired date.
- Episode Number: Episode number within the season.
- Episode Title: Episode title in SonyLiv.
- Pitch Number: Overall pitch number.
- Startup Name: Startup company name.
- Industry: Industry name or type.
- Business Description: Business Description.
- Company Website: Company Website URL.
- Number of Presenters: Number of presenters.
- Male Presenters: Number of male presenters.
- Female Presenters: Number of female presenters.
- Transgender Presenters: Number of transgender/LGBTQ presenters.
- Couple Presenters: Are presenters wife/husband? 1-yes, 0-no.
- Pitchers Average Age: Pitchers average age, categorized as <30 young, 30-50 middle, >50 old.
- Started in: Year in which the startup was started/incorporated.
- Pitchers City: Presenter's town/city.
- Pitchers State: Indian state pitcher hails from.
- Yearly Revenue: Yearly revenue, in lakhs INR, -1 means negative revenue, 0 means pre-revenue.
- Monthly Sales: Total monthly sales, in lakhs.
- Gross Margin: Gross margin/profit of the company, in percentages.
- Net Margin: Net margin/profit of the company, in percentages.
- Original Ask Amount: Original Ask Amount, in lakhs INR.
- Original Offered Equity: Original Offered Equity, in percentages.
- Valuation Requested: Valuation Requested, in lakhs INR.
- Received Offer: Received offer or not, 1-received, 0-not received.
- Accepted Offer: Accepted offer or not, 1-accepted, 0-rejected.
- Total Deal Amount: Total Deal Amount, in lakhs INR.
- Total Deal Equity: Total Deal Equity, in percentages.
- Total Deal Debt: Total Deal Debt, in lakhs INR.
- Debt Interest: Debt interest rate, in percentages.
- Deal Valuation: Deal Valuation, in lakhs INR.
- Number of sharks in the deal: Number of sharks involved in the deal.
- Deal has conditions: Deal has conditions or not?
- Has Patents: Pitcher has Patents? 1-yes, 0-no.
- Additional investment-related fields for individual sharks.

## Project Structure

- **app.py**: The Streamlit application for predicting the outcome of pitches using a trained machine learning model.
- **random_forest_model.joblib**: The trained Random Forest Classifier model for outcome prediction.
- **data_preprocessing.joblib**: Preprocessing steps and encoded columns for the input data.
- **Notebooks**: Jupyter notebooks and scripts used for data analysis and model training.
- **Data**: The dataset (e.g., 'STData_cleaned.csv') used for analysis.

## How to Use the Streamlit App

- Run the Streamlit app using the command: `streamlit run app.py`.
- Input information related to your startup pitch.
- Click the "Make Prediction" button to predict whether your pitch would be accepted or not, along with the confidence score.
- Interpret the results to get insights into the outcome.

For more details, refer to the [GitHub repository](https://github.com/Pranav-V-27/CRISP-DM) for the code and project documentation.

Please feel free to explore the [Streamlit app](https://crisp-dm-final.streamlit.app/) for interactive prediction based on your input.



*Enjoy exploring and predicting the outcomes of your Shark Tank pitches!*

Disclaimer : This project is for educational and experimental purposes only. The generated text may not always make sense, and the model's output should not be taken as factual or accurate. The author is not responsible for any misuse or misinterpretation of the generated content.

Feel free to explore and modify the code for your own text generation experiments!
