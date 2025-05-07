# Import libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import plotly.express as px


# Add a logo and title
col1, col2 = st.columns([1, 3])  # Two columns: 1 for the logo, 3 for the title

with col1:
    # Display the pic
    st.image("logo2.gif", width=False)  
with col2:
    # Add the title with html style
    st.markdown("""
        <h1 style='text-align: left;'>
            <span style='color: #3366FF;'>𝑝𝑦</span>
            <span style='color: #FF5733;'>𝑓𝑖𝑛</span>
        </h1>
    """, unsafe_allow_html=True)

# Add a line below the title and logo
st.markdown("<hr style='border: 2px solid #FF5733; margin-top: 0;'>", unsafe_allow_html=True)





# Sidebar Option Selection
option = st.sidebar.selectbox("What do you want to do?", ('Stock Analysis', 'Currency Converter', 'Stock Prediction'))



# ------#------------#--- STOCK ANALYSIS -#-------------#-------#
if option == 'Stock Analysis':
    stock_symbol = st.text_input("Enter the stock symbol (e.g., AAPL for Apple, MSFT for Microsoft):", placeholder="AAPL, MSFT, GOOGL, AMZN")
    
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("📅 From:", key="start_date", min_value=pd.to_datetime("2000-01-01"), max_value=pd.to_datetime("today"))
    with col_date2:
        end_date = st.date_input("📅 To:", key="end_date", min_value=start_date, max_value=pd.to_datetime("today"))
    
    st.divider()
    
    #function to bring stock data
    def fetch_stock_data(symbol):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return stock, info
        except Exception as e:
            st.error(f"⚠️ Error fetching data: {e}")
            return None, None
    
    if stock_symbol:
        stock, info = fetch_stock_data(stock_symbol)
        if stock and info:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏢 Company Info")
                st.write(f"**Name:** {info.get('longName', 'Not Available')}")
                st.write(f"**Sector:** {info.get('sector', 'Not Available')}")
                st.write(f"**Industry:** {info.get('industry', 'Not Available')}")
                website = info.get('website', None)
                if website:
                    st.markdown(f"[🌐 Visit Website]({website})", unsafe_allow_html=True)
                else:
                    st.write("Website: Not Available")
            with col2:
                st.subheader("📝 Market Data")
                st.write(f"**Current Price:** ${info.get('regularMarketPrice', 'Not Available')}")
                st.write(f"**Market Cap:** ${info.get('marketCap', 'Not Available')}")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'Not Available')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 'Not Available')}")
                st.write(f"**52-Week High:** ${info.get('fiftyTwoWeekHigh', 'Not Available')}")
                st.write(f"**52-Week Low:** ${info.get('fiftyTwoWeekLow', 'Not Available')}")
            
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("📊 Financial Ratios")
                st.write(f"**Debt-to-Equity Ratio:** {info.get('debtToEquity', 'Not Available')}")
                st.write(f"**Return on Equity (ROE):** {info.get('returnOnEquity', 'Not Available')}%")
                st.write(f"**Price-to-Book Ratio:** {info.get('priceToBook', 'Not Available')}")
            
            with col4:
                st.subheader("📈 Technical Indicators")
                data = stock.history(start=start_date, end=end_date)
                if not data.empty:
                    data['50-day SMA'] = data['Close'].rolling(window=50).mean()
                    data['200-day SMA'] = data['Close'].rolling(window=200).mean()
                    data['returns'] = data['Close'].pct_change()  #daily percentage return
                    volatility = data['returns'].std()
                    st.write(f"**50-Day SMA:** {data['50-day SMA'].iloc[-1]:.2f}")
                    st.write(f"**200-Day SMA:** {data['200-day SMA'].iloc[-1]:.2f}")
                    st.write(f"**Volatility (Standard Deviation):** {volatility:.4f}")
                else:
                    st.warning("⚠️ No data available for the selected period.")
            
            if not data.empty:
                st.subheader("📈 Historical Data")
                with st.expander("View Raw Data"):
                    st.dataframe(data[['Close']])
                st.divider()
                st.subheader("💲 Stock Price Chart")
                st.line_chart(data['Close'])
        else:
            st.error("⚠️ Invalid stock symbol or unable to fetch data.")
            
        st.divider()
         
    # -----#---- FINANCIAL STATEMENTS --#-------#
        st.subheader("📜 Financial Statements")

    # Income Statement
        try:
            income_statement = stock.financials
            if not income_statement.empty:
                st.write("💰 Income Statement:")
                with st.expander("View Data"):
                    st.dataframe(income_statement)
            else:
                st.warning("⚠️ Income Statement data not available.")
        except Exception as e:
            st.error(f"Error fetching Income Statement: {e}")

        # Balance Sheet 
        try:
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty:
                st.write("📌 Balance Sheet:")
                with st.expander("View Data"):
                    st.dataframe(balance_sheet)
            else:
                st.warning("⚠️ Balance sheet data is empty or not available.")
        except Exception as e:
            st.error(f"Error fetching Balance Sheet: {e}")

        # Cash Flow Statement
        try:
            cash_flow = stock.cashflow
            if not cash_flow.empty:
                st.write("💸 Cash Flow Statement:")
                with st.expander("View Data"):
                    st.dataframe(cash_flow)
            else:
                st.warning("⚠️ Cash Flow Statement data not available.")
        except Exception as e:
            st.error(f"Error fetching Cash Flow Statement: {e}")



# ---#------------#------ CURRENCY CONVERTER ----##------#-----------#
if option == 'Currency Converter':
    st.title("💱 Currency Converter")

    amount = st.number_input("Enter the amount", min_value=0.01, value=1.0) #min value 0.01 to avoid negative values

    #  currency list
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "SAR", "AED",]

    from_currency = st.selectbox("From Currency", currencies)
    to_currency = st.selectbox("To Currency", currencies)

    API_KEY = "f4c2e7e6751628a148086fcd"  # free API key 1500 requests per month
    api_url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{from_currency}"

    if st.button("Convert"):
        try:
            response = requests.get(api_url)
            data = response.json()

            if response.status_code == 200 and data.get("result") == "success":
                rates = data["conversion_rates"]

                if to_currency in rates:
                    converted_amount = amount * rates[to_currency]
                    st.success(f"{amount} {from_currency} = {converted_amount:.6f} {to_currency}")
                else:
                    st.error(f"⚠️ Exchange rate for {to_currency} not available.")
            else:
                st.error("⚠️ Failed to fetch exchange rates. Please try again.")

        except Exception as e:
            st.error(f"An error occurred during conversion: {e}")

    st.markdown("<hr style='border: 2px solid #FF5733; margin-top: 0;'>", unsafe_allow_html=True)


# ------#--------#------- STOCK PREDICTION --#----------#---------#
if option == 'Stock Prediction':
    
    ## Function to compute the Relative Strength Index (RSI)
    def compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Computes the Relative Strength Index (RSI)."""
        delta = data.diff()                                                                                #the difference between  closing prices.
        gain = delta.where(delta > 0, 0)                                                                   #positive changes in price
        loss = -delta.where(delta < 0, 0)                                                                  #negative changes in price
        avg_gain = gain.rolling(window).mean().replace(0, 1e-10)                                           #rolling mean of gains and losses
        avg_loss = loss.rolling(window).mean().replace(0, 1e-10)
        rs = avg_gain / avg_loss                                                                            #ratio average gain to average loss
        return 100 - (100 / (1 + rs))                                                                       #if rsi > 70 overbought, if rsi < 30 oversold

    st.title("📈 Stock Prediction Model")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL").strip().upper()
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.6, 0.05)

    data = pd.DataFrame()                                                                                     # Initialize an empty DataFrame to save the stock data
    if ticker:
        with st.spinner("Fetching data..."):   
            try:
                stock = yf.Ticker(ticker)                                                                     # bring 10 years of data for the stock
                data = stock.history(period="10y")
            except Exception as e:
                st.error(f"⚠️ Error fetching data: {e}")
                st.stop()

        if data.empty:
            st.error(f"⚠️ No data available for {ticker}")
            st.stop()

        data.index = pd.to_datetime(data.index, utc=True)                                                       # Convert index to datatime format
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")                                # Drop unnecessary columns
        
        # Feature Engineering   
                  
        data["Tomorrow"] = data["Close"].shift(-1)                                                              # Tomorrow:Shift the "Close" prices one day forward
        data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)                                         # Target: 1 if price goes up, 0 if it goes down
        data["RSI"] = compute_rsi(data["Close"], window=14)                                                     # RSI calculation
        
        horizons = [2, 5, 60, 250, 1000]                                                                         # Define the horizons for rolling averages and trends
        predictor_columns = ["Close", "Volume", "Open", "High", "Low", "RSI"]                                    # Predictor columns for the model
        for horizon in horizons:                                                                                 # Calculate rolling averages and trends
            rolling_avg = data["Close"].rolling(horizon).mean()     
            data[f"CLOSE_RATIO_{horizon}"] = data["Close"] / rolling_avg                              # Calculate the ratio of the current close price to the rolling average
            data[f"TREND_{horizon}"] = data["Target"].shift(1).rolling(horizon).sum()                            # Calculate the trend over the horizon
            predictor_columns.extend([f"CLOSE_RATIO_{horizon}", f"TREND_{horizon}"])                             # Add the new features to the predictor columns
        
        data.dropna(inplace=True)  # Drop rows with NaN values 
        
        # Model Training: 200 number of trees in the forest
        model_rf = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1, n_jobs=-1) 
        
        def backtest(df: pd.DataFrame, model: RandomForestClassifier, predictors: list, threshold: float) -> pd.DataFrame: 
            """Performs backtesting on hsistorical stock data."""    
            start = min(1000, len(df) // 2)                                                                                ## Start from 1000 or half the data length
            step = 250                                                                                                      # Step size for backtesting
            all_preds = []                                                                                                 # List to store predictions
            
            progress_bar = st.progress(0)                                                                                   # Progress bar for backtesting
            steps = len(range(start, len(df), step))                                                                        # Total number of steps for backtesting
            
            for i, idx in enumerate(range(start, len(df), step)):  
                train = df.iloc[:idx]                                                                                       # Training set up to the current index
                test = df.iloc[idx:idx + step]                                                                              # Test set for the current iteration
                if test.empty:
                    continue
                
                model.fit(train[predictors], train["Target"])                                                                # Fit the model on the training set
                preds = (model.predict_proba(test[predictors])[:, 1] >= threshold).astype(int)                               # Predict on the test set
                predictions_df = pd.DataFrame({"Target": test["Target"], "Predictions": preds}, index=test.index)            # Create DataFrame for predictions
                all_preds.append(predictions_df)                                                                             # Append predictions to the list
                
                progress_bar.progress((i + 1) / steps)     # Update progress bar
            
            progress_bar.empty()                                                                                             # Clear the progress bar after completion
            return pd.concat(all_preds) if all_preds else pd.DataFrame()                                              # Concatenate all predictions into a single DataFrame
        
        with st.spinner("Training model..."):
            predictions_df = backtest(data, model_rf, predictor_columns, threshold)                                     # Perform backtesting and generate predictions
        
        if predictions_df.empty:
            st.error("⚠️ Not enough data to generate predictions.")
            st.stop()
        
        # Metrics Calculation
        # Calculate precision, recall, F1-score, and ROC-AUC
        precision = precision_score(predictions_df["Target"], predictions_df["Predictions"])
        recall = recall_score(predictions_df["Target"], predictions_df["Predictions"])
        f1 = f1_score(predictions_df["Target"], predictions_df["Predictions"])
        roc_auc = roc_auc_score(predictions_df["Target"], predictions_df["Predictions"])
        
        col1, col2, col3, col4 = st.columns(4)                                                    # Create four columns for metrics display 
        col1.metric("Precision", f"{precision:.2f}")                                              # Display precision
        col2.metric("Recall", f"{recall:.2f}")
        col3.metric("F1-Score", f"{f1:.2f}")
        col4.metric("ROC-AUC", f"{roc_auc:.2f}")
        
   
        # Profit Calculation If the model predicts an increase in price (Predictions == 1), the profit is calculated as the price difference (minus transaction costs).
        try:
            predictions_df["Profit"] = predictions_df.apply( 
                lambda row: (data.loc[row.name, "Tomorrow"] - data.loc[row.name, "Close"]) * 0.999  # Assuming a 0.1% transaction cost 
                if row["Predictions"] == 1 else 0,
                axis=1
            )
            total_profit = predictions_df["Profit"].sum()  # Total profit calculation
            st.metric("Total Profit (after costs)", f"${total_profit:,.2f}")  
        except KeyError as e:
            st.warning(f"⚠️ Profit calculation error: {e}")
   
        
             
        # Display Predictions
        # Plotting actual vs predicted prices
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index[-len(predictions_df):], data["Close"].iloc[-len(predictions_df):], label="Actual Price", color="blue")
        ax.scatter(predictions_df.index, data.loc[predictions_df.index, "Close"], c=predictions_df["Predictions"], cmap="coolwarm", label="Prediction", alpha=0.7)
        ax.set_title("Actual vs Predicted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend() 
        st.pyplot(fig)
        
        

    
      

        # Pie chart for prediction distribution using Plotly
        
        # Count the number of 'Buy' and 'Hold' predictions
        prediction_counts = predictions_df["Predictions"].value_counts()
        labels = ["Buy" if i == 1 else "Hold" for i in prediction_counts.index]
        values = prediction_counts.values

        # Create  pie chart with Plotly
        fig = px.pie(
            names=labels,
            values=values,
            title="Prediction Distribution",
            color_discrete_sequence=["#4CAF50", "#FFC107"],  # (Green for Buy, Yellow for Hold)
            hole=0.3  # Add a hole in the middle to make it look like a donut chart
        )

        # Customize the layout for a modern look
        fig.update_traces(
            textinfo="percent+label",  # Show percentage and label
            pull=[0.1, 0],  # Pull out the "Buy" slice slightly for emphasis
            hoverinfo="label+percent+value",  # Show detailed info on hover
            marker=dict(line=dict(color="#000000", width=2))  # Add a border to slices
        )

        fig.update_layout(
            title_font_size=20,  # Increase title font size
            title_x=0.5,  # Center the title
            margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins
            font=dict(size=14, color="#333333"),  # Customize font size and color
            showlegend=True  # Show legend
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)





# Heatmap for feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pd.DataFrame(model_rf.feature_importances_, index=predictor_columns, columns=["Importance"]).sort_values(by="Importance", ascending=False), annot=True, cmap="Blues")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    
       
    
        # Download results as CSV
        st.download_button(
    label="Download Predictions as CSV",
    data=predictions_df.to_csv(index=False),
    file_name="stock_predictions.csv",
    mime="text/csv"
)
    
    
        # Add a column for price change

    data["Price_Change"] = data["Tomorrow"] - data["Close"]

    # Backtest and generate predictions
    with st.spinner("Training model..."):
        predictions_df = backtest(data, model_rf, predictor_columns, threshold)

    # Add expected price change based on predictions
    predictions_df["Expected_Price_Change"] = predictions_df.apply(
        lambda row: data.loc[row.name, "Price_Change"] if row["Predictions"] == 1 else 0,
        axis=1
    )

    # Display the results
    st.subheader("📈 Predicted Price Changes")
    st.write(predictions_df[["Predictions", "Expected_Price_Change"]])

    # Plot the expected price changes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(predictions_df.index, predictions_df["Expected_Price_Change"], color=["green" if x > 0 else "red" for x in predictions_df["Expected_Price_Change"]])
    ax.set_title("Expected Price Changes Based on Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price Change ($)")
    st.pyplot(fig)
        
        
    
    
    
        # Feature Importance 
    st.markdown("<hr style='border: 2px solid #FF5733; margin-top: 0;'>", unsafe_allow_html=True)
    st.markdown("👨‍💻 Developed by: Elias - 20058836")
