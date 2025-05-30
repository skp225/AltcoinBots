# Import necessary libraries
import re
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, Span, Label, LegendItem, ColumnDataSource, TabPanel, Tabs
from bokeh.layouts import column
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
import random
import glob

# Suppress warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

def convert_to_numeric(value):
    """
    Convert a value to numeric format.
    If the value is already a float, returns the value as it is.
    If the value contains 'K', 'M', or 'B', converts it to a numeric format accordingly.
    :param value: the value to be converted
    :type value: str or float
    :return: the numeric value
    :rtype: float
    """
    if isinstance(value, float):
        return value
    elif isinstance(value, str):
        if 'K' in value:
            return float(re.sub(r'[^\d.]', '', value)) * 1000
        elif 'M' in value:
            return float(re.sub(r'[^\d.]', '', value)) * 1000000
        elif 'B' in value:
            return float(re.sub(r'[^\d.]', '', value)) * 1000000000
        else:
            try:
                return float(re.sub(r'[^\d.]', '', value))
            except:
                return np.nan
    else:
        return np.nan

def load_article_sentiment_data(crypto_name=None):
    """
    Load article sentiment data from the sentiment_data/articles directory.
    
    :param crypto_name: Name of the cryptocurrency to filter sentiment data for (optional)
    :return: DataFrame with sentiment data
    """
    # Define path to sentiment data directory
    sentiment_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "Sentiment", "sentiment_data", "articles")
    
    print(f"Loading article sentiment data from {sentiment_dir}...")
    
    # Initialize an empty DataFrame to store all sentiment data
    all_sentiment_data = pd.DataFrame()
    
    # Check if the directory exists
    if not os.path.exists(sentiment_dir):
        print(f"Warning: Article sentiment data directory not found at {sentiment_dir}")
        return all_sentiment_data
    
    # Get all JSON files in the directory
    all_files = glob.glob(os.path.join(sentiment_dir, "*.json"))
    
    if not all_files:
        print(f"Warning: No article sentiment data files found in {sentiment_dir}")
        return all_sentiment_data
    
    # Process each file
    for file_path in all_files:
        try:
            file_name = os.path.basename(file_path)
            
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Convert to DataFrame
            file_df = pd.DataFrame(articles)
            
            # Skip if empty
            if file_df.empty:
                continue
                
            # Extract date from published field
            if 'published' in file_df.columns:
                file_df['DataUpdateDate'] = pd.to_datetime(file_df['published'], errors='coerce')
            
            # Filter for specific cryptocurrency if provided
            if crypto_name and 'project_name' in file_df.columns:
                file_df = file_df[
                    (file_df['project_name'].str.lower() == crypto_name.lower()) | 
                    (file_df['project_name'].str.contains(crypto_name, case=False, na=False))
                ]
            
            # Extract sentiment metrics
            if 'sentiment' in file_df.columns:
                # Expand sentiment dictionary into separate columns
                sentiment_df = pd.json_normalize(file_df['sentiment'])
                
                # Rename columns to avoid conflicts
                sentiment_df.columns = ['article_' + col for col in sentiment_df.columns]
                
                # Reset index to ensure proper alignment
                file_df = file_df.reset_index(drop=True)
                sentiment_df = sentiment_df.reset_index(drop=True)
                
                # Combine with main DataFrame
                file_df = pd.concat([file_df, sentiment_df], axis=1)
                
                # Drop the original sentiment column
                file_df = file_df.drop(columns=['sentiment'])
            
            # Append to the combined DataFrame
            all_sentiment_data = pd.concat([all_sentiment_data, file_df], ignore_index=True)
            
            print(f"Loaded article sentiment data from {file_name}. Articles: {len(file_df)}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # If we have data, process it
    if not all_sentiment_data.empty:
        # Sort by date
        all_sentiment_data = all_sentiment_data.sort_values('DataUpdateDate')
        
        # Standardize project names
        if 'project_name' in all_sentiment_data.columns:
            # Clean up project names
            all_sentiment_data['project_name'] = all_sentiment_data['project_name'].apply(
                lambda x: x.replace('*Unknown: ', '').replace('*', '') if isinstance(x, str) and '*Unknown:' in x else x
            )
            
            # Map common abbreviations to full names
            name_mapping = {
                'BTC': 'Bitcoin',
                'ETH': 'Ethereum',
                'SOL': 'Solana',
                'XRP': 'Ripple',
                'DOGE': 'Dogecoin',
                'LTC': 'Litecoin',
                'UNI': 'Uniswap',
                'LINK': 'Chainlink',
                'ADA': 'Cardano',
                'DOT': 'Polkadot',
                'AVAX': 'Avalanche',
                'MATIC': 'Polygon'
            }
            
            all_sentiment_data['project_name'] = all_sentiment_data['project_name'].replace(name_mapping)
        
        # Calculate daily average sentiment metrics by project
        if 'project_name' in all_sentiment_data.columns:
            # Convert date to date only (no time)
            all_sentiment_data['date_only'] = all_sentiment_data['DataUpdateDate'].dt.date
            
            # Group by project and date
            sentiment_metrics = ['article_polarity', 'article_subjectivity', 'article_pos', 'article_neu', 'article_neg']
            available_metrics = [col for col in sentiment_metrics if col in all_sentiment_data.columns]
            
            if available_metrics:
                # Calculate daily averages
                daily_sentiment = all_sentiment_data.groupby(['project_name', 'date_only'])[available_metrics].mean().reset_index()
                
                # Convert date back to datetime
                daily_sentiment['DataUpdateDate'] = pd.to_datetime(daily_sentiment['date_only'])
                daily_sentiment = daily_sentiment.drop(columns=['date_only'])
                
                # Calculate rolling averages
                daily_sentiment['sentiment_3d_avg'] = daily_sentiment.groupby('project_name')['article_polarity'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                
                daily_sentiment['sentiment_7d_avg'] = daily_sentiment.groupby('project_name')['article_polarity'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )
                
                # Calculate sentiment momentum (change in sentiment)
                daily_sentiment['sentiment_momentum'] = daily_sentiment.groupby('project_name')['article_polarity'].transform(
                    lambda x: x.diff()
                )
                
                # Replace the original DataFrame with the aggregated one
                all_sentiment_data = daily_sentiment
        
        print(f"Processed article sentiment data. Final shape: {all_sentiment_data.shape}")
    else:
        print("No valid article sentiment data found.")
    
    return all_sentiment_data

def load_trend_data():
    """
    Load trend data from the sentiment_data/trends directory.
    
    :return: DataFrame with trend data
    """
    # Define path to trend data directory
    trend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "Sentiment", "sentiment_data", "trends")
    
    print(f"Loading trend data from {trend_dir}...")
    
    # Initialize an empty DataFrame to store all trend data
    all_trend_data = pd.DataFrame()
    
    # Check if the directory exists
    if not os.path.exists(trend_dir):
        print(f"Warning: Trend data directory not found at {trend_dir}")
        return all_trend_data
    
    # Get all JSON files in the directory
    all_files = glob.glob(os.path.join(trend_dir, "*.json"))
    
    if not all_files:
        print(f"Warning: No trend data files found in {trend_dir}")
        return all_trend_data
    
    # Process each file
    for file_path in all_files:
        try:
            file_name = os.path.basename(file_path)
            
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                trend_data = json.load(f)
            
            # Extract timestamp
            timestamp = pd.to_datetime(trend_data.get('timestamp', ''), errors='coerce')
            
            # Extract trending projects
            trending_projects = trend_data.get('trending_projects', [])
            
            # Convert to DataFrame
            if trending_projects:
                trend_df = pd.DataFrame(trending_projects)
                
                # Add timestamp
                trend_df['DataUpdateDate'] = timestamp
                
                # Append to the combined DataFrame
                all_trend_data = pd.concat([all_trend_data, trend_df], ignore_index=True)
                
                print(f"Loaded trend data from {file_name}. Projects: {len(trending_projects)}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # If we have data, process it
    if not all_trend_data.empty:
        # Sort by date
        all_trend_data = all_trend_data.sort_values('DataUpdateDate')
        
        # Clean up project names
        all_trend_data['name'] = all_trend_data['name'].apply(
            lambda x: x.replace('*Unknown: ', '').replace('*', '') if isinstance(x, str) and '*Unknown:' in x else x
        )
        
        # Map common abbreviations to full names
        name_mapping = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'SOL': 'Solana',
            'XRP': 'Ripple',
            'DOGE': 'Dogecoin',
            'LTC': 'Litecoin',
            'UNI': 'Uniswap',
            'LINK': 'Chainlink',
            'ADA': 'Cardano',
            'DOT': 'Polkadot',
            'AVAX': 'Avalanche',
            'MATIC': 'Polygon'
        }
        
        all_trend_data['name'] = all_trend_data['name'].replace(name_mapping)
        
        # Calculate trend metrics
        # Group by date and project to get daily mentions
        daily_mentions = all_trend_data.groupby(['DataUpdateDate', 'name'])['mentions'].sum().reset_index()
        
        # Pivot to get projects as columns
        trend_pivot = daily_mentions.pivot(index='DataUpdateDate', columns='name', values='mentions').reset_index()
        
        # Fill NaN values with 0 (no mentions)
        trend_pivot = trend_pivot.fillna(0)
        
        # Calculate total mentions per day
        total_mentions = daily_mentions.groupby('DataUpdateDate')['mentions'].sum().reset_index()
        total_mentions = total_mentions.rename(columns={'mentions': 'total_mentions'})
        
        # Merge back to get mention percentages
        trend_pivot = pd.merge(trend_pivot, total_mentions, on='DataUpdateDate', how='left')
        
        # Calculate percentage of total mentions for each project
        for col in trend_pivot.columns:
            if col not in ['DataUpdateDate', 'total_mentions']:
                trend_pivot[f'{col}_pct'] = trend_pivot[col] / trend_pivot['total_mentions'] * 100
        
        # Replace the original DataFrame with the processed one
        all_trend_data = trend_pivot
        
        print(f"Processed trend data. Final shape: {all_trend_data.shape}")
    else:
        print("No valid trend data found.")
    
    return all_trend_data

def load_market_data():
    """
    Load Fear & Greed Index and Stock/ETF data for market sentiment analysis.
    
    :return: DataFrame with market sentiment data
    """
    # Define paths to data files
    fear_greed_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "GeneralMarketAnalysis", "FearGreedData", "fear_greed_historical.csv")
    
    stock_etf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "GeneralMarketAnalysis", "StockETFData")
    
    # Load Fear & Greed Index data
    try:
        print("Loading Fear & Greed Index data...")
        fear_greed_df = pd.read_csv(fear_greed_path)
        fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'])
        fear_greed_df = fear_greed_df.rename(columns={'date': 'DataUpdateDate'})
        fear_greed_df = fear_greed_df[['DataUpdateDate', 'value', 'value_classification']]
        print(f"Fear & Greed data loaded successfully. Shape: {fear_greed_df.shape}")
    except Exception as e:
        print(f"Error loading Fear & Greed data: {str(e)}")
        fear_greed_df = pd.DataFrame(columns=['DataUpdateDate', 'value', 'value_classification'])
    
    # Load key Stock/ETF data (focusing on Bitcoin ETFs and crypto-related stocks)
    key_etfs = ['IBIT', 'GBTC', 'BITQ', 'MSTR']
    stock_etf_data = {}
    
    for etf in key_etfs:
        try:
            etf_path = os.path.join(stock_etf_dir, f"{etf}_historical_data.csv")
            if os.path.exists(etf_path):
                print(f"Loading {etf} data...")
                etf_df = pd.read_csv(etf_path)
                etf_df['Date'] = pd.to_datetime(etf_df['Date'])
                etf_df = etf_df.rename(columns={'Date': 'DataUpdateDate'})
                etf_df = etf_df[['DataUpdateDate', 'Close', 'Volume']]
                etf_df = etf_df.rename(columns={'Close': f'{etf}_close', 'Volume': f'{etf}_volume'})
                stock_etf_data[etf] = etf_df
                print(f"{etf} data loaded successfully. Shape: {etf_df.shape}")
        except Exception as e:
            print(f"Error loading {etf} data: {str(e)}")
    
    # Merge all market data
    market_data = fear_greed_df
    
    for etf, data in stock_etf_data.items():
        if not data.empty:
            market_data = pd.merge(market_data, data, on='DataUpdateDate', how='outer')
    
    # Fill missing values
    market_data = market_data.sort_values('DataUpdateDate')
    
    # Forward fill for missing dates (use previous day's values)
    market_data = market_data.fillna(method='ffill')
    
    # Calculate additional market metrics
    if 'IBIT_close' in market_data.columns:
        market_data['IBIT_change_pct'] = market_data['IBIT_close'].pct_change()
    
    if 'GBTC_close' in market_data.columns:
        market_data['GBTC_change_pct'] = market_data['GBTC_close'].pct_change()
    
    # Calculate market trend indicators (7-day moving averages)
    if 'value' in market_data.columns:
        market_data['fg_7d_avg'] = market_data['value'].rolling(window=7).mean()
    
    if 'IBIT_close' in market_data.columns:
        market_data['IBIT_7d_avg'] = market_data['IBIT_close'].rolling(window=7).mean()
    
    # Fill any remaining NaN values with median values
    market_data = market_data.fillna(market_data.median())
    
    return market_data

def calculate_growth_score(crypto_data):
    """
    Calculate a growth score for each cryptocurrency based on various metrics.
    Higher score indicates higher likelihood of price increase.
    Only includes cryptocurrencies with data for the most recent date and at least 30 days of history.
    
    :param crypto_data: DataFrame containing cryptocurrency data
    :return: DataFrame with growth scores
    """
    # Get the most recent date in the dataset
    most_recent_date = crypto_data['DataUpdateDate'].max()
    print(f"Most recent date in dataset: {most_recent_date}")
    
    # Calculate the date 30 days before the most recent date
    min_required_date = most_recent_date - timedelta(days=30)
    print(f"Minimum required date for analysis: {min_required_date}")
    
    # Get unique cryptocurrencies
    cryptos = crypto_data['name'].unique()
    print(f"Total unique cryptocurrencies: {len(cryptos)}")
    
    # Create a DataFrame to store scores
    scores_df = pd.DataFrame(columns=['name', 'symbol', 'growth_score', 'current_price', 'market_cap', 'data_days'])
    
    # Counter for cryptocurrencies with sufficient data
    valid_crypto_count = 0
    
    for crypto in cryptos:
        # Get data for this cryptocurrency
        crypto_subset = crypto_data[crypto_data['name'] == crypto].copy()
        
        # Sort by date
        crypto_subset = crypto_subset.sort_values('DataUpdateDate')
        
        # Check if this cryptocurrency has data for the most recent date
        has_recent_data = most_recent_date in crypto_subset['DataUpdateDate'].values
        
        # Calculate the date range of available data
        if len(crypto_subset) >= 2:
            first_date = crypto_subset['DataUpdateDate'].min()
            date_range = (crypto_subset['DataUpdateDate'].max() - first_date).days
            has_sufficient_history = first_date <= min_required_date
        else:
            date_range = 0
            has_sufficient_history = False
        
        # Skip if not enough data points or doesn't have recent data or insufficient history
        if len(crypto_subset) < 5 or not has_recent_data or not has_sufficient_history:
            continue
        
        valid_crypto_count += 1
        
        # Get the latest data point
        latest_data = crypto_subset.iloc[-1]
        
        # Calculate metrics for scoring
        
        # 1. Recent price trend (last 5 days)
        recent_prices = crypto_subset['current_price'].tail(5).values
        if len(recent_prices) >= 2:
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        else:
            price_trend = 0
            
        # 2. Distance from ATH (closer to ATH might indicate strength)
        ath_proximity = 1 + (latest_data['ath_change_percentage'] / 100)  # Convert negative percentage to a positive factor
        
        # 3. Recent volume relative to market cap
        volume_to_mcap = latest_data['total_volume'] / latest_data['market_cap'] if latest_data['market_cap'] > 0 else 0
        
        # 4. Recent market cap growth
        recent_mcap = crypto_subset['market_cap'].tail(5).values
        if len(recent_mcap) >= 2:
            mcap_growth = (recent_mcap[-1] - recent_mcap[0]) / recent_mcap[0] if recent_mcap[0] > 0 else 0
        else:
            mcap_growth = 0
            
        # 5. Price change momentum (acceleration of price changes)
        if len(crypto_subset) >= 3:
            price_changes = crypto_subset['price_change_percentage_24h'].tail(3).values
            momentum = np.mean(price_changes) if not np.isnan(price_changes).any() else 0
        else:
            momentum = 0
            
        # Calculate composite score (weights can be adjusted)
        growth_score = (
            price_trend * 0.35 +
            ath_proximity * 0.15 +
            volume_to_mcap * 0.15 +
            mcap_growth * 0.25 +
            momentum * 0.1
        )
        
        # Add to scores DataFrame
        scores_df = pd.concat([scores_df, pd.DataFrame({
            'name': [latest_data['name']],
            'symbol': [latest_data['symbol']],
            'growth_score': [growth_score],
            'current_price': [latest_data['current_price']],
            'market_cap': [latest_data['market_cap']],
            'data_days': [date_range]
        })], ignore_index=True)
    
    print(f"Cryptocurrencies with sufficient data (recent + 30 days history): {valid_crypto_count}")
    
    # Sort by growth score in descending order
    scores_df = scores_df.sort_values('growth_score', ascending=False).reset_index(drop=True)
    
    return scores_df

def train_prediction_model(crypto_data, crypto_name, market_data=None, sentiment_data=None, trend_data=None):
    """
    Train a prediction model for a specific cryptocurrency
    
    :param crypto_data: DataFrame containing data for all cryptocurrencies
    :param crypto_name: Name of the cryptocurrency to train model for
    :param market_data: DataFrame containing market sentiment data (optional)
    :param sentiment_data: DataFrame containing project-specific sentiment data (optional)
    :param trend_data: DataFrame containing trend data (optional)
    :return: Trained model, scalers, and processed data
    """
    # Filter data for the specific cryptocurrency
    data = crypto_data[crypto_data['name'] == crypto_name].copy()
    
    # Sort by date
    data = data.sort_values('DataUpdateDate')
    
    # Select relevant features
    feature_columns = [
        'current_price', 'market_cap', 'total_volume', 
        'price_change_percentage_24h', 'market_cap_change_percentage_24h',
        'ath_change_percentage', 'atl_change_percentage'
    ]
    
    # Handle missing values
    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].interpolate(method='linear', limit_direction='both')
    
    # Check if we have enough data
    if len(data) < 10:
        return None, None, None, None, None, None
    
    # Add additional features for better prediction
    # Add rolling means for different windows
    data['price_rolling_3d'] = data['current_price'].rolling(window=3).mean()
    data['price_rolling_7d'] = data['current_price'].rolling(window=7).mean()
    
    # Add price momentum (rate of change)
    data['price_momentum_3d'] = data['current_price'].pct_change(periods=3)
    data['price_momentum_7d'] = data['current_price'].pct_change(periods=7)
    
    # Add volatility (standard deviation of returns)
    data['volatility_7d'] = data['current_price'].pct_change().rolling(window=7).std()
    
    # Fill NaN values created by rolling windows
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    # Add market data features if available
    market_feature_columns = []
    if market_data is not None and not market_data.empty:
        # Merge with market data
        data = pd.merge(data, market_data, on='DataUpdateDate', how='left')
        
        # Add Fear & Greed Index if available
        if 'value' in data.columns:
            market_feature_columns.append('value')  # Fear & Greed Index value
            market_feature_columns.append('fg_7d_avg')  # 7-day average of Fear & Greed Index
        
        # Add ETF data if available
        for col in data.columns:
            if any(etf in col for etf in ['IBIT', 'GBTC', 'BITQ', 'MSTR']) and col.endswith(('_close', '_change_pct', '_7d_avg')):
                market_feature_columns.append(col)
        
        # Fill any missing market data with forward fill then backward fill
        for col in market_feature_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
    
    # Add project-specific sentiment data if available
    sentiment_feature_columns = []
    if sentiment_data is not None and not sentiment_data.empty:
        # Filter sentiment data for this specific cryptocurrency
        crypto_sentiment = sentiment_data[sentiment_data['project_name'] == crypto_name].copy()
        
        if not crypto_sentiment.empty:
            print(f"Found project-specific sentiment data for {crypto_name}. Shape: {crypto_sentiment.shape}")
            
            # Merge with main data
            sentiment_cols = [col for col in crypto_sentiment.columns if col not in ['project_name']]
            data = pd.merge(data, crypto_sentiment[['DataUpdateDate'] + sentiment_cols], 
                           on='DataUpdateDate', how='left')
            
            # Fill missing sentiment values with neutral values
            for col in sentiment_cols:
                if col in data.columns:
                    # For polarity and similar metrics, use 0 as neutral
                    if 'polarity' in col or 'momentum' in col or 'sentiment' in col:
                        data[col] = data[col].fillna(0)
                    # For subjectivity, use 0.5 as neutral
                    elif 'subjectivity' in col:
                        data[col] = data[col].fillna(0.5)
                    # For counts, use 0
                    else:
                        data[col] = data[col].fillna(0)
                    
                    sentiment_feature_columns.append(col)
        else:
            print(f"No project-specific sentiment data found for {crypto_name}")
    
    # Add trend data if available
    trend_feature_columns = []
    if trend_data is not None and not trend_data.empty:
        # Check if this cryptocurrency is in the trend data
        if crypto_name in trend_data.columns:
            print(f"Found trend data for {crypto_name}")
            
            # Merge with main data
            trend_cols = [crypto_name, f'{crypto_name}_pct']
            if all(col in trend_data.columns for col in trend_cols):
                data = pd.merge(data, trend_data[['DataUpdateDate'] + trend_cols], 
                               on='DataUpdateDate', how='left')
                
                # Fill missing trend values with 0 (no mentions)
                for col in trend_cols:
                    if col in data.columns:
                        data[col] = data[col].fillna(0)
                        trend_feature_columns.append(col)
                
                # Add rolling averages for trend data
                if crypto_name in data.columns:
                    data[f'{crypto_name}_3d_avg'] = data[crypto_name].rolling(window=3).mean()
                    data[f'{crypto_name}_7d_avg'] = data[crypto_name].rolling(window=7).mean()
                    trend_feature_columns.extend([f'{crypto_name}_3d_avg', f'{crypto_name}_7d_avg'])
                
                # Add momentum for trend data (change in mentions)
                if crypto_name in data.columns:
                    data[f'{crypto_name}_momentum'] = data[crypto_name].diff()
                    trend_feature_columns.append(f'{crypto_name}_momentum')
                
                # Fill NaN values created by rolling windows
                for col in trend_feature_columns:
                    if col in data.columns:
                        data[col] = data[col].fillna(0)
        else:
            print(f"No trend data found for {crypto_name}")
    
    # Combine all feature columns
    extended_features = feature_columns + [
        'price_rolling_3d', 'price_rolling_7d',
        'price_momentum_3d', 'price_momentum_7d',
        'volatility_7d'
    ] + market_feature_columns + sentiment_feature_columns + trend_feature_columns
    
    # Prepare features and target
    features = data[extended_features].copy()
    target = data['current_price'].copy()
    
    # Convert date to numeric for model input
    data['date_numeric'] = (data['DataUpdateDate'] - data['DataUpdateDate'].min()).dt.days
    features['date_numeric'] = data['date_numeric']
    
    # Scale features
    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    # Scale target
    target_scaler = MinMaxScaler()
    target_scaled = target.values.reshape(-1, 1)
    target_scaled = target_scaler.fit_transform(target_scaled)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)
    
    # Build ensemble model (combining RandomForest and GradientBoosting)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Train models
    rf_model.fit(x_train, y_train.ravel())
    gb_model.fit(x_train, y_train.ravel())
    
    # Create ensemble model (as a dictionary of models)
    model = {
        'rf': rf_model,
        'gb': gb_model
    }
    
    return model, feature_scaler, target_scaler, features, target, data

def predict_future_prices(model, feature_scaler, target_scaler, features, data, market_data=None, sentiment_data=None, trend_data=None, future_days=7):
    """
    Predict future prices for a cryptocurrency
    
    :param model: Trained model (dictionary of models)
    :param feature_scaler: Scaler for features
    :param target_scaler: Scaler for target
    :param features: Feature DataFrame
    :param data: Original data DataFrame
    :param market_data: DataFrame containing market sentiment data (optional)
    :param sentiment_data: DataFrame containing project-specific sentiment data (optional)
    :param trend_data: DataFrame containing trend data (optional)
    :param future_days: Number of days to predict into the future (default: 7 days)
    :return: DataFrame with future predictions
    """
    if model is None:
        return None
        
    # Get the last date in the data
    last_date = data['DataUpdateDate'].max()
    
    # Create future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    
    # Create a DataFrame for future predictions
    future_df = pd.DataFrame()
    future_df['DataUpdateDate'] = future_dates
    future_df['date_numeric'] = (future_df['DataUpdateDate'] - data['DataUpdateDate'].min()).dt.days
    
    # Get the last row of features to use as a starting point
    last_features = features.iloc[-1:].copy()
    
    # Initialize lists to store predictions
    all_predictions = []
    
    # Get historical volatility to add realistic variations to predictions
    historical_volatility = data['current_price'].pct_change().std()
    
    # Get current market sentiment if available
    current_market_sentiment = None
    if market_data is not None and not market_data.empty and 'value' in market_data.columns:
        # Get the most recent Fear & Greed value
        recent_market_data = market_data.sort_values('DataUpdateDate', ascending=False).iloc[0]
        current_market_sentiment = recent_market_data['value']
        
        # Add market data to future predictions
        for col in market_data.columns:
            if col != 'DataUpdateDate' and col in last_features.columns:
                # For future predictions, use the most recent market data value
                last_features[col] = recent_market_data[col]
    
    # Get current project-specific sentiment if available
    current_project_sentiment = None
    crypto_name = data['name'].iloc[0] if 'name' in data.columns else None
    
    if sentiment_data is not None and not sentiment_data.empty and crypto_name:
        # Filter sentiment data for this specific cryptocurrency
        crypto_sentiment = sentiment_data[sentiment_data['project_name'] == crypto_name]
        
        if not crypto_sentiment.empty:
            # Get the most recent sentiment score
            recent_sentiment = crypto_sentiment.sort_values('DataUpdateDate', ascending=False).iloc[0]
            
            # Get the sentiment polarity (main sentiment score)
            if 'article_polarity' in recent_sentiment:
                current_project_sentiment = recent_sentiment['article_polarity']
            
            # Add sentiment data to future predictions
            for col in recent_sentiment.index:
                if col != 'DataUpdateDate' and col != 'project_name' and col in last_features.columns:
                    last_features[col] = recent_sentiment[col]
    
    # Get current trend data if available
    current_trend_mentions = None
    if trend_data is not None and not trend_data.empty and crypto_name and crypto_name in trend_data.columns:
        # Get the most recent trend data
        recent_trend = trend_data.sort_values('DataUpdateDate', ascending=False).iloc[0]
        
        # Get the number of mentions
        if crypto_name in recent_trend:
            current_trend_mentions = recent_trend[crypto_name]
        
        # Add trend data to future predictions
        for col in recent_trend.index:
            if col != 'DataUpdateDate' and col in last_features.columns:
                last_features[col] = recent_trend[col]
    
    # Make predictions for each future day
    for i in range(future_days):
        # Update the date_numeric for this prediction
        last_features['date_numeric'] = future_df['date_numeric'].iloc[i]
        
        # Scale the features
        scaled_features = feature_scaler.transform(last_features)
        
        # Make predictions with both models in the ensemble
        rf_prediction_scaled = model['rf'].predict(scaled_features).reshape(-1, 1)
        gb_prediction_scaled = model['gb'].predict(scaled_features).reshape(-1, 1)
        
        # Average the predictions
        avg_prediction_scaled = (rf_prediction_scaled + gb_prediction_scaled) / 2
        
        # Inverse transform to get the actual price
        prediction = target_scaler.inverse_transform(avg_prediction_scaled)[0][0]
        
        # Apply sentiment adjustments
        sentiment_adjustment_applied = False
        
        # 1. Apply market sentiment adjustment if available
        if current_market_sentiment is not None:
            # Fear & Greed Index ranges from 0-100
            # 0-25: Extreme Fear, 25-50: Fear, 50-75: Greed, 75-100: Extreme Greed
            if current_market_sentiment >= 75:  # Extreme Greed - market might be overheated
                # Reduce prediction by up to 50% based on how extreme the greed is
                # The higher the greed index, the more we reduce the prediction
                # At 75, reduction is minimal; at 100, reduction is maximum (50%)
                adjustment_factor = 1.0 - ((current_market_sentiment - 75) / 50)  # Max 50% reduction at 100
                prediction = prediction * adjustment_factor
                print(f"Day {i+1}: Applied market sentiment adjustment (Extreme Greed): {adjustment_factor:.2f}")
                sentiment_adjustment_applied = True
            elif current_market_sentiment <= 25:  # Extreme Fear - market might be oversold
                # Increase prediction by up to 5% based on how extreme the fear is
                adjustment_factor = 1.0 + ((25 - current_market_sentiment) / 500)  # Max 5% increase at 0
                prediction = prediction * adjustment_factor
                print(f"Day {i+1}: Applied market sentiment adjustment (Extreme Fear): {adjustment_factor:.2f}")
                sentiment_adjustment_applied = True
        
        # 2. Apply project-specific sentiment adjustment if available and no market sentiment adjustment was applied
        if current_project_sentiment is not None and not sentiment_adjustment_applied:
            # Project sentiment typically ranges from -1 (very negative) to 1 (very positive)
            if current_project_sentiment > 0.5:  # Very positive sentiment
                # Increase prediction by up to 10% based on sentiment strength
                adjustment_factor = 1.0 + (current_project_sentiment - 0.5) / 5  # Max 10% increase at sentiment = 1
                prediction = prediction * adjustment_factor
                print(f"Day {i+1}: Applied project sentiment adjustment (Positive): {adjustment_factor:.2f}")
                sentiment_adjustment_applied = True
            elif current_project_sentiment < -0.5:  # Very negative sentiment
                # Decrease prediction by up to 15% based on sentiment strength
                adjustment_factor = 1.0 + (current_project_sentiment + 0.5) / 3.33  # Max 15% decrease at sentiment = -1
                prediction = prediction * adjustment_factor
                print(f"Day {i+1}: Applied project sentiment adjustment (Negative): {adjustment_factor:.2f}")
                sentiment_adjustment_applied = True
        
        # 3. Apply trend data adjustment if available and no sentiment adjustment was applied
        if current_trend_mentions is not None and not sentiment_adjustment_applied:
            # If the cryptocurrency has a high number of mentions, it might indicate increased interest
            if current_trend_mentions > 10:  # High number of mentions
                # Adjust prediction based on mention count (higher mentions = higher adjustment)
                # Cap the adjustment to avoid extreme values
                adjustment_factor = 1.0 + min(current_trend_mentions / 100, 0.15)  # Max 15% increase
                prediction = prediction * adjustment_factor
                print(f"Day {i+1}: Applied trend data adjustment (High Mentions): {adjustment_factor:.2f}")
        
        # Add some realistic variation based on historical volatility
        # This helps avoid the flat line predictions
        variation = np.random.normal(0, historical_volatility * prediction * 0.5)
        prediction = max(0, prediction + variation)  # Ensure price doesn't go negative
        
        # Store the prediction
        all_predictions.append(prediction)
        
        # Update the last_features for the next iteration
        # Update current_price
        last_features['current_price'] = prediction
        
        # Update rolling means
        if 'price_rolling_3d' in last_features.columns:
            if i >= 2:
                last_features['price_rolling_3d'] = np.mean(all_predictions[-3:])
            else:
                recent_prices = list(data['current_price'].tail(3-i).values) + all_predictions
                last_features['price_rolling_3d'] = np.mean(recent_prices[-3:])
                
        if 'price_rolling_7d' in last_features.columns:
            if i >= 6:
                last_features['price_rolling_7d'] = np.mean(all_predictions[-7:])
            else:
                recent_prices = list(data['current_price'].tail(7-i).values) + all_predictions
                last_features['price_rolling_7d'] = np.mean(recent_prices[-7:])
        
        # Update momentum features
        if 'price_momentum_3d' in last_features.columns:
            if i >= 3:
                # Avoid division by zero
                if all_predictions[-4] > 0:
                    last_features['price_momentum_3d'] = (all_predictions[-1] / all_predictions[-4] - 1)
                else:
                    last_features['price_momentum_3d'] = 0  # Default to 0 if denominator is zero
            else:
                idx = 3 - i
                old_price = data['current_price'].iloc[-idx]
                # Avoid division by zero
                if old_price > 0:
                    last_features['price_momentum_3d'] = (prediction / old_price - 1)
                else:
                    last_features['price_momentum_3d'] = 0  # Default to 0 if denominator is zero
                
        if 'price_momentum_7d' in last_features.columns:
            if i >= 7:
                # Avoid division by zero
                if all_predictions[-8] > 0:
                    last_features['price_momentum_7d'] = (all_predictions[-1] / all_predictions[-8] - 1)
                else:
                    last_features['price_momentum_7d'] = 0  # Default to 0 if denominator is zero
            else:
                idx = 7 - i
                old_price = data['current_price'].iloc[-idx]
                # Avoid division by zero
                if old_price > 0:
                    last_features['price_momentum_7d'] = (prediction / old_price - 1)
                else:
                    last_features['price_momentum_7d'] = 0  # Default to 0 if denominator is zero
        
        # Update volatility
        if 'volatility_7d' in last_features.columns:
            if i >= 6:
                # Avoid division by zero
                pct_changes = []
                for j in range(i-6, i):
                    if all_predictions[j] > 0:
                        pct_changes.append(all_predictions[j+1]/all_predictions[j]-1)
                    else:
                        pct_changes.append(0)  # Default to 0 if denominator is zero
                
                if pct_changes:  # Make sure we have at least one valid change
                    last_features['volatility_7d'] = np.std(pct_changes)
                else:
                    last_features['volatility_7d'] = historical_volatility
            else:
                # Use historical volatility for initial predictions
                last_features['volatility_7d'] = historical_volatility
    
    # Add predictions to future_df
    future_df['predicted_price'] = all_predictions
    
    # Add market sentiment information if available
    if current_market_sentiment is not None:
        future_df['market_sentiment'] = current_market_sentiment
        
        # Add sentiment classification
        if current_market_sentiment <= 25:
            sentiment_class = "Extreme Fear"
        elif current_market_sentiment <= 50:
            sentiment_class = "Fear"
        elif current_market_sentiment <= 75:
            sentiment_class = "Greed"
        else:
            sentiment_class = "Extreme Greed"
            
        future_df['sentiment_classification'] = sentiment_class
    
    # Add project-specific sentiment information if available
    if current_project_sentiment is not None:
        future_df['project_sentiment'] = current_project_sentiment
        
        # Add sentiment classification
        if current_project_sentiment <= -0.5:
            project_sentiment_class = "Very Negative"
        elif current_project_sentiment <= 0:
            project_sentiment_class = "Negative"
        elif current_project_sentiment <= 0.5:
            project_sentiment_class = "Positive"
        else:
            project_sentiment_class = "Very Positive"
            
        future_df['project_sentiment_classification'] = project_sentiment_class
    
    # Add trend data information if available
    if current_trend_mentions is not None:
        future_df['trend_mentions'] = current_trend_mentions
    
    return future_df

def create_interactive_chart(data, future_data, crypto_name):
    """
    Create an interactive chart for a cryptocurrency
    
    :param data: Historical data
    :param future_data: Future predictions
    :param crypto_name: Name of the cryptocurrency
    :return: Bokeh figure
    """
    # Filter historical data to only show the last 14 days
    last_date = data['DataUpdateDate'].max()
    start_date = last_date - pd.Timedelta(days=14)
    filtered_data = data[data['DataUpdateDate'] >= start_date].copy()
    
    # Create figure
    p = figure(
        title=f'{crypto_name} Price Prediction (Last 14 Days + 7 Day Forecast)',
        x_axis_label='Date',
        y_axis_label='Price',
        width=1000,
        height=500,
        x_axis_type="datetime"
    )
    
    # Add historical price line (last 14 days only)
    source_hist = ColumnDataSource(data={
        'date': filtered_data['DataUpdateDate'],
        'price': filtered_data['current_price'],
        'formatted_date': filtered_data['DataUpdateDate'].dt.strftime('%Y-%m-%d')
    })
    
    p.line(
        'date', 'price',
        source=source_hist,
        legend_label='Historical Price (Last 14 Days)',
        line_color='blue',
        line_width=2
    )
    
    # Add future predictions if available
    if future_data is not None:
        # Prepare tooltip data
        tooltip_data = {
            'date': future_data['DataUpdateDate'],
            'price': future_data['predicted_price'],
            'formatted_date': future_data['DataUpdateDate'].dt.strftime('%Y-%m-%d')
        }
        
        # Add market sentiment information if available
        if 'market_sentiment' in future_data.columns:
            tooltip_data['market_sentiment'] = future_data['market_sentiment']
            tooltip_data['sentiment_classification'] = future_data['sentiment_classification']
        
        # Add project-specific sentiment information if available
        if 'project_sentiment' in future_data.columns:
            tooltip_data['project_sentiment'] = future_data['project_sentiment']
            tooltip_data['project_sentiment_classification'] = future_data['project_sentiment_classification']
        
        # Add trend data information if available
        if 'trend_mentions' in future_data.columns:
            tooltip_data['trend_mentions'] = future_data['trend_mentions']
        
        source_future = ColumnDataSource(data=tooltip_data)
        
        # Add prediction line
        p.line(
            'date', 'price',
            source=source_future,
            legend_label='Predicted Price',
            line_color='red',
            line_dash='dashed',
            line_width=2
        )
        
        # Add distinct points for each prediction day
        p.circle(
            'date', 'price',
            source=source_future,
            size=8,
            color='red',
            alpha=0.8
        )
    
    # Add hover tool
    hover = HoverTool()
    
    # Configure tooltip based on available data
    tooltips = [
        ('Date', '@formatted_date'),
        ('Price', '@price{0.00000000}')
    ]
    
    if future_data is not None:
        if 'market_sentiment' in future_data.columns:
            tooltips.append(('Market Sentiment', '@sentiment_classification (@market_sentiment{0.0})'))
        
        if 'project_sentiment' in future_data.columns:
            tooltips.append(('Project Sentiment', '@project_sentiment_classification (@project_sentiment{0.00})'))
        
        if 'trend_mentions' in future_data.columns:
            tooltips.append(('Trend Mentions', '@trend_mentions{0}'))
    
    hover.tooltips = tooltips
    p.add_tools(hover)
    
    # Style the plot
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

def main():
    print("Loading cryptocurrency data...")
    # Load the data using dynamic path resolution
    try:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dailymergedv3.csv")
        if not os.path.exists(data_file):
            # Try sample file if full data file doesn't exist
            data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dailymergedv3sample.csv")
        
        data = pd.read_csv(data_file)
        print(f"Data loaded successfully from {data_file}. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    print(f"Columns: {data.columns.tolist()}")
    print(f"Sample of first few rows:")
    print(data.head(2))
    
    # Check for unique cryptocurrencies
    if 'name' in data.columns:
        unique_cryptos = data['name'].unique()
        print(f"Number of unique cryptocurrencies: {len(unique_cryptos)}")
        print(f"Sample of cryptocurrencies: {unique_cryptos[:5] if len(unique_cryptos) > 5 else unique_cryptos}")
    else:
        print("Warning: 'name' column not found in data")
        return
    
    # Clean the data
    print("\nCleaning and preprocessing data...")
    
    # Try different date formats for DataUpdateDate
    date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']
    for date_format in date_formats:
        try:
            # Try to convert a sample to check if format works
            sample = data['DataUpdateDate'].iloc[0] if not data.empty else ""
            pd.to_datetime(sample, format=date_format)
            print(f"Detected date format: {date_format}")
            # If successful, convert all dates
            data['DataUpdateDate'] = pd.to_datetime(data['DataUpdateDate'], format=date_format, errors='coerce')
            break
        except:
            continue
    
    # Convert other date columns with flexible parsing
    data['ath_date'] = pd.to_datetime(data['ath_date'], errors='coerce')
    data['atl_date'] = pd.to_datetime(data['atl_date'], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = [
        'current_price', 'market_cap', 'total_volume',
        'high_24h', 'low_24h', 'price_change_24h',
        'price_change_percentage_24h', 'market_cap_change_24h',
        'market_cap_change_percentage_24h', 'ath', 'ath_change_percentage',
        'atl', 'atl_change_percentage'
    ]
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].apply(convert_to_numeric)
    
    # Load market data (Fear & Greed Index and Stock/ETF data)
    print("\nLoading market sentiment data...")
    market_data = load_market_data()
    
    # Load article sentiment data
    print("\nLoading article sentiment data...")
    sentiment_data = load_article_sentiment_data()
    
    # Load trend data
    print("\nLoading trend data...")
    trend_data = load_trend_data()
    
    # Ask user for specific cryptocurrencies to analyze
    print("\nWhich cryptocurrencies would you like to analyze?")
    print("Enter cryptocurrency names separated by commas, or leave blank to use top 5 by growth score")
    
    user_input = input("Enter cryptocurrency names (e.g., Bitcoin, Ethereum, Solana): ")
    
    # Process user input
    if user_input.strip():
        # User provided specific cryptocurrencies
        selected_cryptos = [name.strip() for name in user_input.split(',')]
        print(f"\nSelected cryptocurrencies: {selected_cryptos}")
        
        # Verify that the selected cryptocurrencies exist in the data
        valid_cryptos = []
        for crypto in selected_cryptos:
            if crypto in unique_cryptos:
                valid_cryptos.append(crypto)
            else:
                # Try case-insensitive match
                matches = [c for c in unique_cryptos if c.lower() == crypto.lower()]
                if matches:
                    valid_cryptos.append(matches[0])
                else:
                    print(f"Warning: {crypto} not found in the dataset and will be skipped")
        
        if not valid_cryptos:
            print("No valid cryptocurrencies selected. Using top 5 by growth score instead.")
            # Calculate growth scores to find top cryptocurrencies
            growth_scores = calculate_growth_score(data)
            valid_cryptos = growth_scores['name'].head(5).tolist()
    else:
        # No input, use top 5 by growth score
        print("\nNo specific cryptocurrencies selected. Using top 5 by growth score.")
        growth_scores = calculate_growth_score(data)
        valid_cryptos = growth_scores['name'].head(5).tolist()
    
    print(f"\nAnalyzing the following cryptocurrencies: {valid_cryptos}")
    
    # Create output directory for HTML files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"crypto_predictions_{timestamp}.html"
    
    # Create tabs for each cryptocurrency
    tabs = []
    
    # Process each selected cryptocurrency
    for crypto_name in valid_cryptos:
        print(f"\nProcessing {crypto_name}...")
        
        # Train prediction model
        model, feature_scaler, target_scaler, features, target, crypto_data = train_prediction_model(
            data, crypto_name, market_data, sentiment_data, trend_data
        )
        
        if model is None:
            print(f"Insufficient data for {crypto_name}. Skipping.")
            continue
        
        # Make predictions
        future_data = predict_future_prices(
            model, feature_scaler, target_scaler, features, crypto_data, 
            market_data, sentiment_data, trend_data
        )
        
        # Create chart
        p = create_interactive_chart(crypto_data, future_data, crypto_name)
        
        # Add to tabs
        tabs.append(TabPanel(child=p, title=crypto_name))
    
    # Create the final output with tabs
    if tabs:
        tabs_layout = Tabs(tabs=tabs)
        output_file(output_file_name)
        show(tabs_layout)
        print(f"\nPrediction charts saved to {output_file_name}")
    else:
        print("\nNo valid cryptocurrencies could be processed. Please try again with different selections.")

if __name__ == "__main__":
    main()
