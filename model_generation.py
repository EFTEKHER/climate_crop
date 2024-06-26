import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import joblib

# Load the dataset
data = pd.read_csv("./climate-ds.csv")


sample_data = data

# Drop unnecessary columns ('Unnamed: 0', 'Area')
sample_data.drop(['Unnamed: 0', 'Area'], axis=1, inplace=True)

# Label Encoding for 'Item' column
label_encoder = LabelEncoder()
sample_data['Item'] = label_encoder.fit_transform(sample_data['Item'])

# Define features and target
features = ['Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
target = 'hg/ha_yield'

X = sample_data[features]
y = sample_data[target]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

# Define models and hyperparameters
models = {
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGB': XGBRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'AdaBoost': AdaBoostRegressor()
}

param_grids = {
    'Decision Tree': {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 10, 20]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [10, 15, 20]},
    'XGB': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'Extra Trees': {'n_estimators': [50, 100, 150], 'max_depth': [10, 15, 20]},
    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
}

results = []
best_estimators = {}

for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_estimators[model_name] = best_model
    y_pred = best_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results.append({'Model': model_name, 'R2 Score': r2, 'MAE': mae})

results_df = pd.DataFrame(results)

# Save the models, label encoder, and results with joblib using compression
joblib.dump((best_estimators, label_encoder, results_df), 'models.pkl', compress=7)
