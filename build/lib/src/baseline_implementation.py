import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
STATE_SETTINGS = {
    'texas': {'min_alerts': 100, 'test_size': 0.2},
    'michigan': {'min_alerts': 50, 'test_size': 0.2},
    'hawaii': {'min_alerts': 20, 'test_size': 0.3}
}

# Enhanced feature list
FEATURE_COLS = [
    'event_type', 'pre_max_outage', 'affected_counties',
    'WINDTAG', 'HAILTAG', 'duration_hours', 'alert_severity'
]

def load_and_preprocess(file_path):
    """Enhanced data loading with proper aggregation"""
    print("Loading data...")
    df = pd.read_csv(file_path, parse_dates=['ISSUED', 'EXPIRED'])
    
    # Calculate alert duration
    df['duration_hours'] = (df['EXPIRED'] - df['ISSUED']).dt.total_seconds() / 3600
    
    # Create severity score
    df['alert_severity'] = df['WINDTAG'].fillna(0) + df['HAILTAG'].fillna(0)
    
    # Filter and clean
    df = df[df['outage_count'].notna()]
    df['state'] = df['state'].str.lower().str.strip()
    
    return df.sort_values('ISSUED')

def create_aggregated_features(df):
    """Create event-based features matching expected format"""
    print("Creating aggregated features...")
    
    # Group by weather events (same WFO within 6 hours)
    df['event_group'] = (df.groupby(['state', 'WFO'])['ISSUED']
                        .diff().dt.total_seconds() > 6*3600).cumsum()
    
    features = []
    for (state, group_id), group in tqdm(df.groupby(['state', 'event_group']),
                                       desc="Processing events"):
        if len(group) < 1:
            continue
            
        # Calculate features
        duration = group['duration_hours'].max()
        max_outage = group['outage_count'].max()
        total_outages = group['outage_count'].sum()
        
        features.append({
            'state': state,
            'event_type': group['PHENOM'].iloc[0],
            'pre_max_outage': max_outage,
            'affected_counties': len(group['county'].unique()),
            'WINDTAG': group['WINDTAG'].max(),
            'HAILTAG': group['HAILTAG'].max(),
            'duration_hours': duration,
            'alert_severity': group['alert_severity'].max(),
            'target': total_outages  # Using cumulative outages as target
        })
    
    return pd.DataFrame(features)

def build_model_pipelines():
    """Configure models with proper preprocessing"""
    numeric_features = ['pre_max_outage', 'affected_counties', 
                       'WINDTAG', 'HAILTAG', 'duration_hours', 'alert_severity']
    categorical_features = ['event_type']
    
    preprocessor = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    return {
        'RandomForest': make_pipeline(
            preprocessor,
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        'kNN': make_pipeline(
            preprocessor,
            KNeighborsRegressor(n_neighbors=3)
        ),
        'XGBoost': make_pipeline(
            preprocessor,
            xgb.XGBRegressor(n_estimators=100, random_state=42)
        )
    }

def evaluate_models(features, models):
    """Enhanced evaluation with proper time-based splitting"""
    results = []
    
    for state in STATE_SETTINGS:
        state_data = features[features['state'] == state]
        if len(state_data) < STATE_SETTINGS[state]['min_alerts']:
            continue
            
        # Time-based split
        split_idx = int(len(state_data) * (1 - STATE_SETTINGS[state]['test_size']))
        X_train = state_data[FEATURE_COLS].iloc[:split_idx]
        X_test = state_data[FEATURE_COLS].iloc[split_idx:]
        y_train = state_data['target'].iloc[:split_idx]
        y_test = state_data['target'].iloc[split_idx:]
        
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results.append({
                    'state': state,
                    'model': model_name,
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'samples': len(state_data)
                })
            except Exception as e:
                print(f"Error with {model_name} in {state}: {str(e)}")
                continue
    
    return pd.DataFrame(results)

def main():
    # Load and process data
    data = load_and_preprocess('data/outage_count.csv')
    features = create_aggregated_features(data)
    
    # Build and evaluate models
    models = build_model_pipelines()
    results = evaluate_models(features, models)
    
    # Save and display results
    if not results.empty:
        results.to_csv('improved_results.csv', index=False)
        print("\nFinal Results:")
        print(results.groupby(['state', 'model']).mean(numeric_only=True))
    else:
        print("No valid results produced")

if __name__ == "__main__":
    main()