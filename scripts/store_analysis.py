import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
# Function to load datasets
def load_data(train_path, store_path):
    try:
        train_data = pd.read_csv(train_path)
        store_data = pd.read_csv(store_path)
        logging.info("Datasets successfully loaded.")
        return train_data, store_data
    except FileNotFoundError as e:
        logging.error(f"Error loading datasets: {e}")
        raise

# Function to merge datasets
def merge_data(train_data, store_data):
    logging.info("Merging train and store datasets.")
    return pd.merge(train_data, store_data, on='Store', how='left')

# Function to preprocess data
def preprocess_data(data):
    # Convert 'Date' to datetime
    logging.info("Converting 'Date' column to datetime format.")
    data['Date'] = pd.to_datetime(data['Date'])

    # Add time-based features
    logging.info("Adding time-based features.")
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Day'] = data['Date'].dt.day
    data['DayOfYear'] = data['Date'].dt.dayofyear

    # Handle missing values
    logging.info("Handling missing values.")
    data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
    data['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    data['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    data['Promo2SinceWeek'].fillna(0, inplace=True)
    data['Promo2SinceYear'].fillna(0, inplace=True)
    data['PromoInterval'].fillna('None', inplace=True)

    # Remove outliers in Sales
    logging.info("Removing outliers in 'Sales'.")
    Q1 = data['Sales'].quantile(0.25)
    Q3 = data['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['Sales'] >= lower_bound) & (data['Sales'] <= upper_bound)]
    
    return data

# Function to plot promotion distribution
def plot_promo_distribution(data):
    logging.info("Checking distribution of promotions in training set.")
    promo_distribution = data['Promo'].value_counts(normalize=True)
    print("Promo Distribution in Training Set:")
    print(promo_distribution)

# Function to analyze holiday effects
def plot_holiday_effects(data):
    logging.info("Analyzing sales behavior around holidays.")
    holiday_sales = data.groupby('StateHoliday')['Sales'].mean()
    holiday_sales.plot(kind='bar', color='orange')
    plt.title('Average Sales During Holidays')
    plt.xlabel('Holiday Type')
    plt.ylabel('Average Sales')
    plt.show()

# Function to analyze seasonal effects
def plot_seasonal_effects(data):
    logging.info("Analyzing seasonal effects on sales.")
    monthly_sales = data.groupby('Month')['Sales'].mean()
    monthly_sales.plot(kind='line', marker='o', color='purple')
    plt.title('Average Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.grid(True)
    plt.show()

# Function to analyze correlation between sales and customers
def plot_correlation(data):
    logging.info("Analyzing correlation between sales and customers.")
    correlation = data['Sales'].corr(data['Customers'])
    print(f"Correlation between Sales and Customers: {correlation:.2f}")
    sns.scatterplot(x=data['Customers'], y=data['Sales'])
    plt.title('Correlation between Customers and Sales')
    plt.xlabel('Customers')
    plt.ylabel('Sales')
    plt.show()

# Function to analyze the effect of promotions on sales
def plot_promo_effect(data):
    logging.info("Analyzing the effect of promotions on sales.")
    promo_effect = data.groupby('Promo')['Sales'].mean()
    promo_effect.plot(kind='bar', color='cyan')
    plt.title('Effect of Promotions on Sales')
    plt.xlabel('Promotion')
    plt.ylabel('Average Sales')
    plt.show()

# Function to analyze which stores should have promos deployed
def plot_store_promo_effect(data):
    logging.info("Analyzing which stores should have promos deployed.")
    store_promo_effect = data.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
    store_promo_effect.plot(kind='bar', figsize=(10, 6))
    plt.title('Sales by Store Type and Promotion')
    plt.ylabel('Average Sales')
    plt.show()

# Function to analyze customer behavior during store opening and closing times
def plot_opening_closing_effect(data):
    logging.info("Analyzing customer behavior during store open/close times.")
    opening_sales = data.groupby('Open')['Sales'].mean()
    opening_sales.plot(kind='bar', color='magenta')
    plt.title('Average Sales by Store Open/Close Status')
    plt.xlabel('Open')
    plt.ylabel('Average Sales')
    plt.show()

# Function to analyze the effect of assortment type on sales
def plot_assortment_effect(data):
    logging.info("Analyzing the effect of assortment type on sales.")
    assortment_sales = data.groupby('Assortment')['Sales'].mean()
    assortment_sales.plot(kind='bar', color='green')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.show()

# Function to analyze the effect of competitor distance on sales
def plot_competition_distance_effect(data):
    logging.info("Analyzing the effect of competitor distance on sales.")
    sns.scatterplot(x=data['CompetitionDistance'], y=data['Sales'])
    plt.title('Sales vs. Competition Distance')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()

# Function to analyze the effect of competitor openings on sales
def plot_competitor_openings_effect(data):
    logging.info("Analyzing effect of competitor openings on sales.")
    data['CompetitionOpenSince'] = data['CompetitionOpenSinceYear'] + data['CompetitionOpenSinceMonth'] / 12
    sns.lineplot(data=data, x='CompetitionOpenSince', y='Sales', ci=None)
    plt.title('Sales Over Time Since Competitor Opened')
    plt.xlabel('Years Since Competitor Opened')
    plt.ylabel('Sales')
    plt.show()