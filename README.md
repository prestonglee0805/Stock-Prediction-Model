# Stock-Prediction-Model
This project is a Long Short-Term Memory (LSTM) model designed to predict stock prices using sequential data. It leverages historical stock price data to forecast the next day's stock price. The project includes preprocessing, model training, and evaluation.

<h2><u>Features</u></h2>

- Fetches stock data using yfinance.

- Normalizes data using MinMaxScaler for improved model performance.

- Splits data into training and testing sets (80% training, 20% testing).

- Creates sequences of stock prices for use in LSTM.

<h2><u>Model Architecture</u></h2>

- Built with Keras Sequential API.

- Two LSTM layers for capturing sequential dependencies.

- Dropout layers for regularization.

- Dense output layer for single-value predictions.

<h2><u>Training Configuration</u></h2>

- Optimizer: Adam.

- Loss Function: Mean Squared Error (MSE).

- Early stopping to prevent overfitting.

<h2><u>Model Evaluation</u></h2>

- Evaluates predictions using MSE and Root Mean Squared Error (RMSE).

- Compares predicted prices with actual prices.

<h2><u>Installation</u></h2>

Ensure you have Python 3.7 or higher installed. 

Required Libraries

- numpy

- pandas

- yfinance

- tensorflow

- scikit-learn

- matplotlib

<h2><u>How To Use</u></h2>

1. Run the Notebook:
Open the Stock_Predictor.ipynb file in Jupyter Notebook or JupyterLab.

2. Enter Stock Symbol:
Enter ticker symbol when prompted to fetch data for dsired stock  

3. Enter Time Span: 
Enter the starting and ending date of data collection in format `YYYY-MM-DD`

4. Preprocess Data:
The notebook will fetch stock data, normalize it, and create training/testing sequences.

5. Train the Model:
Train the LSTM model using the preprocessed data. The training process includes validation and early stopping.

6. Predict Stock Prices:
Use the trained model to predict the next day's stock price based on the last 100 days of data.

<h2><u>Evaluate Model performance</u></h2>

- Compare predicted prices with true values.

- Calculate MSE and RMSE to assess accuracy.

<h2><u>Model Architecture</u></h2>

Input Shape: (time_steps, 1)

- LSTM Layer (units=100, return_sequences=True)
- Dropout Layer (rate=0.2)
- LSTM Layer (units=100, return_sequences=False)
- Dropout Layer (rate=0.2)
- Dense Layer (units=1)

<h2><u>Evaluation Metrics</u></h2>

Mean Squared Error (MSE):

Measures the average squared difference between predicted and actual values.

Root Mean Squared Error (RMSE):

Provides the error in the same units as the target variable, making it more interpretable.
