# Stock Price Prediction 

## Introduction
This project predicts the stock prices of a company using historical stock data. Two approaches are used:
1. **Classification Model**: Predicting if the next day's closing price will be higher than today's.
2. **Regression Model**: Predicting the exact closing price using features like moving averages and previous day's closing price.

## Project Structure

## Dependencies
The following Python libraries are required to run the project:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `Jupyter Notebook`

- 
## Features 
1. **Exploratory Data Analysis (EDA)**
Stock price history visualization.
Adding target variables like tomorrow (next day's close) and Target (binary classification if tomorrow's price is higher).
2. **Stock Price Prediction - Classification Model**
- **Algorithm:** Random Forest Classifier
- **Features used:** open, high, low, close, volume
- **Target:** Whether the closing price increases the next day.
- **Evaluation Metrics:** Precision, Classification Report
- **Precision Score:** 56% (can vary depending on data split)
3. **Stock Price Prediction - Regression Model**
- **Algorithm:** Random Forest Regressor
- **Features used:** Moving averages (25-day, 50-day, 200-day), and previous day's close.
- **Target:** The closing price of the stock.
- **Evaluation Metrics:** Mean Squared Error (MSE)
- **MSE:** 3.57 (can vary depending on data split)
5. **Visualization**
- **Stock closing price history.**
- **Predicted vs Actual stock prices (for both classification and regression models).**

## Results
### Classification Model:
The Random Forest Classifier predicts whether the stock price will go up the next day.
- **Precision Score:** 56% (for the test data used).
### Regression Model:
The Random Forest Regressor predicts the actual closing price.
- **Mean Squared Error (MSE):** 3.57
- **Screenshots:** Actual vs Predicted Stock Prices - Regression Model

## Conclusion
This project demonstrates two different approaches to stock price prediction: a classification model that predicts whether the stock will rise and a regression model that predicts the actual closing price. Both models use historical stock data and provide insights into stock market behavior.
