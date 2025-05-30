import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and prepare data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['DataUpdateDate'])
    df.set_index('date', inplace=True)
    return df

# Calculate moving averages
def calculate_moving_averages(df, window=7):
    df['ma_7'] = df['market_cap'].rolling(window=window).mean()
    df['ma_14'] = df['market_cap'].rolling(window=14).mean()
    df['ma_30'] = df['market_cap'].rolling(window=30).mean()
    return df

# Calculate market cap change percentages
def calculate_market_cap_changes(df):
    df['market_cap_change_pct'] = df['market_cap'].pct_change()
    return df

# Prepare features for models
def prepare_features(df):
    features = df[['ma_7', 'ma_14', 'ma_30',
                   'market_cap_change_pct',
                   'price_change_percentage_24h']]
    target = df['market_cap']
    return features, target

# Handle missing values
def handle_missing_values(df):
    df = df.dropna(subset=['market_cap'])
    df = df.dropna(subset=['ma_7', 'ma_14', 'ma_30', 
                          'market_cap_change_pct', 
                          'price_change_percentage_24h'])
    return df

# Train moving average crossover model
def train_moving_average_model(df):
    df = calculate_moving_averages(df)
    df = calculate_market_cap_changes(df)
    df = handle_missing_values(df)
    features, target = prepare_features(df)
    
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(train_features, train_target)
    
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_target, predictions)
    print(f"Moving Average Model MSE: {mse}")
    return model

# Train decision tree model
def train_decision_tree_model(df):
    df = calculate_moving_averages(df)
    df = calculate_market_cap_changes(df)
    df = handle_missing_values(df)
    features, target = prepare_features(df)
    
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor()
    model.fit(train_features, train_target)
    
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_target, predictions)
    print(f"Decision Tree Model MSE: {mse}")
    return model

# Evaluate a list of projects
def evaluate_project_list(df, project_list):
    results = []
    unique_names = df['name'].unique()
    
    for project in project_list:
        if project.lower() in [n.lower() for n in unique_names]:
            project_data = df[df['name'].str.lower() == project.lower()]
            project_data = handle_missing_values(project_data)
            
            if not project_data.empty:
                ma_model = train_moving_average_model(df)
                dt_model = train_decision_tree_model(df)
                
                ma_prediction = ma_model.predict(project_data[['ma_7', 'ma_14', 'ma_30',
                                                              'market_cap_change_pct',
                                                              'price_change_percentage_24h']])
                dt_prediction = dt_model.predict(project_data[['ma_7', 'ma_14', 'ma_30',
                                                              'market_cap_change_pct',
                                                              'price_change_percentage_24h']])
                
                results.append({
                    'project': project,
                    'short_term_prediction': ma_prediction[0],
                    'long_term_prediction': dt_prediction[0],
                })
            else:
                results.append({
                    'project': project,
                    'status': 'No valid data available'
                })
        else:
            results.append({
                'project': project,
                'status': 'Project not found in database'
            })
    
    return results

# Main function
def main():
    csv_path = "Data/CSV/dailymergedv3.csv"
    df = load_data(csv_path)
    
    # Train models
    ma_model = train_moving_average_model(df)
    dt_model = train_decision_tree_model(df)
    
    # Example usage: Predict for a specific project
    project_name = "ADAMANT Messenger"
    project_data = df[df['name'] == project_name]
    project_data = handle_missing_values(project_data)
    
    if not project_data.empty:
        ma_prediction = ma_model.predict(project_data[['ma_7', 'ma_14', 'ma_30',
                                                      'market_cap_change_pct',
                                                      'price_change_percentage_24h']])
        dt_prediction = dt_model.predict(project_data[['ma_7', 'ma_14', 'ma_30',
                                                      'market_cap_change_pct',
                                                      'price_change_percentage_24h']])
        
        print(f"Predicted market cap using Moving Average Model: ${ma_prediction[0]:.2f}")
        print(f"Predicted market cap using Decision Tree Model: ${dt_prediction[0]:.2f}")
    else:
        print(f"No valid data available for {project_name}")
    
    # Evaluate a list of projects
    project_list = input("Enter a list of project names (comma-separated): ").split(',')
    project_list = [p.strip() for p in project_list]
    
    results = evaluate_project_list(df, project_list)
    
    print("\nProject Evaluations:")
    print("---------------------")
    for result in results:
        if 'status' in result:
            print(f"{result['project']}: {result['status']}")
        else:
            print(f"Project: {result['project']}")
            print(f"Short-term prediction: ${result['short_term_prediction']:.2f}")
            print(f"Long-term prediction: ${result['long_term_prediction']:.2f}")
            print("---------------------")

if __name__ == "__main__":
    main()
