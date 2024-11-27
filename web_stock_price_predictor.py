import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Title of the app
st.title("Stock Price Predictor App")

# User input for the stock ticker symbol
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define date range for downloading stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start, end)

# Display the downloaded stock data
st.subheader("Stock Data")
st.write(google_data)

# Check if stock data is valid
if google_data.empty:
    st.error("Failed to fetch stock data. Please check the stock ID.")
else:
    # Determine the column to use for 'Close' prices
    close_column = 'Close' if 'Close' in google_data.columns else 'Adj Close'
    
    if close_column not in google_data.columns:
        st.error("Neither 'Close' nor 'Adj Close' column exists in the data.")
    else:
        # Load pre-trained model
        try:
            model = tf.keras.models.load_model('./lstm_model.h5')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Splitting data for training and testing
        splitting_len = int(len(google_data) * 0.7)
        x_test = google_data[splitting_len:]

        # Moving average plots
        def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
            fig = plt.figure(figsize=figsize)
            plt.plot(values, 'Orange', label='Moving Average')
            plt.plot(full_data[close_column], 'b', label='Original Close Price')
            if extra_data:
                plt.plot(extra_dataset, label='Additional Data')
            plt.legend()
            return fig

        # Adding moving averages and plotting
        st.subheader("Original Close Price and MA for 250 Days")
        google_data['MA_for_250_days'] = google_data[close_column].rolling(250).mean()
        st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

        st.subheader("Original Close Price and MA for 200 Days")
        google_data['MA_for_200_days'] = google_data[close_column].rolling(200).mean()
        st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

        st.subheader("Original Close Price and MA for 100 Days")
        google_data['MA_for_100_days'] = google_data[close_column].rolling(100).mean()
        st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

        st.subheader("MA for 100 Days vs MA for 250 Days")
        st.pyplot(
            plot_graph(
                (15, 6),
                google_data['MA_for_100_days'],
                google_data,
                1,
                google_data['MA_for_250_days'],
            )
        )

        # Scaling and preparing data for the model
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[[close_column]])

        x_data, y_data = [], []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Making predictions
        predictions = model.predict(x_data)

        # Inverse scaling for predictions and test data
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        # Combine predictions and actual values for visualization
        plotting_data = pd.DataFrame(
            {
                'original_test_data': inv_y_test.reshape(-1),
                'predictions': inv_pre.reshape(-1),
            },
            index=google_data.index[splitting_len + 100:],
        )

        st.subheader("Original Values vs Predicted Values")
        st.write(plotting_data)

        st.subheader("Original Close Price vs Predicted Close Price")
        fig = plt.figure(figsize=(15, 6))
        plt.plot(
            pd.concat([google_data[close_column][:splitting_len + 100], plotting_data], axis=0)
        )
        plt.legend(["Data - Not Used", "Original Test Data", "Predicted Test Data"])
        st.pyplot(fig)
