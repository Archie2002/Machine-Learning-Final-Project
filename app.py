import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.set_page_config(layout="wide")
st.title("Time Series Forecasting App - Retail/Warehouse Sales")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Combine YEAR and MONTH into a single date column
    df["ds"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")

    # Convert relevant columns to numeric
    for col in ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select forecasting target
    target_col = st.selectbox("Select a metric to forecast", ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"])

    df = df[["ds", target_col]].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=["ds", target_col])
    df = df.groupby("ds").sum().asfreq("MS")
    df = df.rename(columns={target_col: "y"})

    # Extra cleaning
    df["y"] = df["y"].astype(float)
    df = df.dropna(subset=["y"])
    df = df[df["y"] > 0]  # optional: remove this if 0s are meaningful

    st.success(f"\u2705 Final cleaned monthly row count: {len(df)}")
    st.write(df.head())
    st.write("Remaining NaNs in y:", df["y"].isna().sum())

    # Decomposition
    st.markdown("## \U0001F50D Seasonal Decomposition")
    model_type = st.radio("Select decomposition type", ["additive", "multiplicative"])

    if len(df.dropna()) >= 24:
        try:
            result = seasonal_decompose(df["y"], model=model_type, period=12)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
            result.observed.plot(ax=ax1, title='Observed')
            result.trend.plot(ax=ax2, title='Trend')
            result.seasonal.plot(ax=ax3, title='Seasonality')
            result.resid.plot(ax=ax4, title='Residual')
            plt.tight_layout(h_pad=2.0)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not perform decomposition: {e}")
    else:
        st.warning("\u26A0 You need at least 24 monthly data points for decomposition.")

    # Forecasting
    st.markdown("## Forecasting")
    model_choice = st.selectbox("Choose a forecasting model", ["ARIMA", "Prophet", "LSTM"])

    if len(df.dropna()) >= 24:
        train = df.iloc[:-12]
        test = df.iloc[-12:]
        forecast = None

        try:
            if model_choice == "ARIMA":
                model = ARIMA(train["y"], order=(5, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
                forecast.index = test.index


            elif model_choice == "Prophet":
                prophet_df = train.reset_index()[["ds", "y"]]
                model = Prophet()
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=12, freq='MS')
                forecast_df = model.predict(future)
                forecast = forecast_df.set_index("ds")["yhat"][-12:]

            elif model_choice == "LSTM":
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[["y"]])
                X, y_lstm = [], []
                for i in range(12, len(scaled_data)):
                    X.append(scaled_data[i-12:i, 0])
                    y_lstm.append(scaled_data[i, 0])
                X, y_lstm = np.array(X), np.array(y_lstm)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y_lstm, epochs=50, verbose=0)

                inputs = scaled_data[-24:]
                X_test = []
                for i in range(12):
                    X_test.append(inputs[i:i+12, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                predictions = model.predict(X_test)
                forecast = scaler.inverse_transform(predictions).flatten()
                forecast = pd.Series(forecast, index=test.index)

            # Plot
            st.subheader(f"Forecast vs Actual - {model_choice}")
            plt.figure(figsize=(12, 5))
            plt.plot(train.index, train["y"], label="Train")
            plt.plot(test.index, test["y"], label="Actual")
            plt.plot(test.index, forecast, label="Forecast", linestyle="--")
            plt.legend()
            st.pyplot(plt)

            # Evaluation Metrics
            rmse = round(np.sqrt(mean_squared_error(test["y"], forecast)), 2)
            mae = round(mean_absolute_error(test["y"], forecast), 2)
            mape = round(mean_absolute_percentage_error(test["y"], forecast) * 100, 2)
            mse = round(mean_squared_error(test["y"], forecast), 2)

            st.markdown("### Evaluation Metrics")
            st.write(f"**RMSE:** {rmse}")
            st.write(f"**MAE:** {mae}")
            st.write(f"**MAPE:** {mape}%")
            st.write(f"**MSE:** {mse}")

        except Exception as e:
            st.error(f"{model_choice} model failed: {e}")

    else:
        st.warning("\u26A0 You need at least 24 data points for reliable forecasting.")

st.markdown("---")
st.markdown("Thank you")
