Time Series Forecasting with CNN and Fibonacci Sequence

This script forecasts future values for a time series dataset using a CNN model with a unique feature engineering approach.

Requirements:

pandas
numpy
datetime
tensorflow
sklearn
How to Use:

Data Preparation:

Replace '' with the actual filename and path to your CSV file containing the time series data.
Ensure your data has a "date" column in a format that can be parsed by pd.to_datetime (e.g., YYYY-MM-DD).
Script Execution:

Bash

python time_series_forecasting_cnn_fibonacci.py
Output:

The script performs a grid search to find the best hyperparameters for the model. It prints the best parameters and the corresponding validation loss.
The script then trains a final model with the best hyperparameters and predicts values for the next days_ahead days (currently set to 7) after the last data point in your dataset.
The predicted values for each day are displayed in a format with applied constraints (e.g., values between certain ranges).
Additionally, the script saves two CSV files:
A CSV containing the predicted values for the next days_ahead days.
A CSV containing the final model predictions on the test set (useful for further analysis).
Notes:

The script uses a standard CNN architecture. You can explore more advanced architectures based on your specific problem.
The script incorporates a Fibonacci sequence as an additional feature. This might not be optimal for all types of time series data. Experiment with different feature engineering techniques.
Modify days_ahead to change the number of days for future predictions.
Disclaimer:

This script provides a basic framework for time series forecasting with a CNN and Fibonacci sequence. You might need to adjust or extend it based on your specific data and requirements.
