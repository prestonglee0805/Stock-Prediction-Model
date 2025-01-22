# Stock-Prediction-Model
This project is a Long Short-Term Memory (LSTM) model designed to predict stock prices using sequential data. It leverages historical stock price data to forecast the next day's stock price. The project includes preprocessing, model training, and evaluation.

#Features

Data Preprocessing:

Fetches stock data using yfinance.

Normalizes data using MinMaxScaler for improved model performance.

Splits data into training and testing sets (80% training, 20% testing).

Creates sequences of stock prices for use in LSTM.

Model Architecture:

Built with Keras Sequential API.

Two LSTM layers for capturing sequential dependencies.

Dropout layers for regularization.

Dense output layer for single-value predictions.

Training Configuration:

Optimizer: Adam.

Loss Function: Mean Squared Error (MSE).

Early stopping to prevent overfitting.

Model Evaluation:

Evaluates predictions using MSE and Root Mean Squared Error (RMSE).

Compares predicted prices with actual prices.

Installation

Prerequisites

Ensure you have Python 3.7 or higher installed. Install the required libraries using:

pip install -r requirements.txt

Required Libraries

numpy

pandas

yfinance

tensorflow

scikit-learn

matplotlib

How to Use

Run the Notebook:
Open the Stock_Predictor.ipynb file in Jupyter Notebook or JupyterLab.

Enter Stock Symbol:
Modify the input to fetch data for your desired stock. Example:

stock_symbol = "AAPL"

Preprocess Data:
The notebook will fetch stock data, normalize it, and create training/testing sequences.

Train the Model:
Train the LSTM model using the preprocessed data. The training process includes validation and early stopping.

Predict Stock Prices:
Use the trained model to predict the next day's stock price based on the last 100 days of data.

Evaluate Model Performance:

Compare predicted prices with true values.

Calculate MSE and RMSE to assess accuracy.

Model Architecture

Input Shape: (time_steps, 1)

- LSTM Layer (units=100, return_sequences=True)
- Dropout Layer (rate=0.2)
- LSTM Layer (units=100, return_sequences=False)
- Dropout Layer (rate=0.2)
- Dense Layer (units=1)

Evaluation Metrics

Mean Squared Error (MSE):

Measures the average squared difference between predicted and actual values.

Root Mean Squared Error (RMSE):

Provides the error in the same units as the target variable, making it more interpretable.

Example Results

True Prices (y_true): [125.34, 126.21, 127.15, ...]
Predicted Prices: [124.89, 126.04, 127.35, ...]

Mean Squared Error: 0.0024
Root Mean Squared Error: 0.049

Troubleshooting

Kernel Issues in Jupyter Notebook:

Restart the kernel or reselect the Python environment.

Input Handling:

If using input(), replace it with a hardcoded value to avoid issues in Jupyter.

Dimension Errors:

Ensure input data is reshaped to (samples, time_steps, features) for LSTM compatibility.

Future Enhancements

Implement hyperparameter tuning.

Extend the model to predict multiple days ahead.

Add support for external factors like trading volume or market indices.

License

This project is licensed under the MIT License.

Acknowledgments

Keras and TensorFlow for the deep learning framework.

Yahoo Finance API for stock price data.

Community resources for guidance on LSTM implementations.
