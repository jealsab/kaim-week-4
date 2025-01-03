import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def predict(train,test):
# Load the train dataset
    # train = pd.read_csv('../data/train.csv')
    # test = pd.read_csv('../data/test.csv')

    # 1. Convert 'StateHoliday' to numeric
    holiday_mapping = {'a': 1, 'b': 2, 'c': 3, '0': 0}
    train['StateHoliday'] = train['StateHoliday'].replace(holiday_mapping).astype(int)
    test['StateHoliday'] = test['StateHoliday'].replace(holiday_mapping).astype(int)

    # 2. Extract datetime features (Date -> Weekday, Weekend, Days to Holiday, etc.)
    def extract_datetime_features(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Weekday'] = df['Date'].dt.weekday
        df['IsWeekend'] = df['Date'].dt.weekday >= 5
        df['DaysToHoliday'] = (df['Date'] - pd.to_datetime('2015-01-01')).dt.days
        df['DaysAfterHoliday'] = (df['Date'] - pd.to_datetime('2015-01-01')).dt.days
        df['MonthStart'] = (df['Date'].dt.day <= 10).astype(int)
        df['MonthMid'] = ((df['Date'].dt.day > 10) & (df['Date'].dt.day <= 20)).astype(int)
        df['MonthEnd'] = (df['Date'].dt.day > 20).astype(int)
        df['MonthOfYear'] = df['Date'].dt.month
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        return df

    # Apply datetime feature extraction to both train and test datasets
    train = extract_datetime_features(train)
    test = extract_datetime_features(test)

    # 3. Define the numerical columns to scale
    numerical_cols = ['Open', 'Promo', 'DaysToHoliday', 'DaysAfterHoliday', 
                    'MonthStart', 'MonthMid', 'MonthEnd', 'MonthOfYear', 'WeekOfYear']

    # 4. Split the dataset into features (X) and target (y)
    X = train[numerical_cols]
    y = train['Sales']  # Assuming 'Sales' is your target variable

    # Split the train data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Create the model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),               # StandardScaler for numerical features
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Random Forest Regressor
    ])

    # 6. Train the model
    pipeline.fit(X_train, y_train)

    # 7. Make predictions and evaluate
    y_pred = pipeline.predict(X_val)

    # 8. Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_val, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Optionally, you can use cross-validation to evaluate the model more robustly
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f'Cross-validated MSE: {-cv_scores.mean()}')

    # Now you can also make predictions on the test dataset
    X_test = test[numerical_cols]
    test_predictions = pipeline.predict(X_test)

    # Optionally save the predictions to a CSV file
    test['Predictions'] = test_predictions
    test[['Id', 'Predictions']].to_csv('predictions.csv', index=False)
