import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

with open('ads_export.json', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

df = df.dropna(subset=['price', 'price_v', 'norm_price'])

df['description'] = df['description'].fillna('No description')

df = df[df['category'].str.startswith('immobilier-vente')]

def extract_property_type(row):
    if row['category'] == 'immobilier-vente-villa':
        return 'Villa'
    elif row['category'] == 'immobilier-vente-terrain':
        return 'Terrain'
    elif row['category'] in ['immobilier-vente-appartement', 'immobilier-vente']:
        match = re.search(r'\b[Ff](0?[1-8])\b', row['title'])
        if match:
            return f'F{match.group(1).lstrip("0")}'
        else:
            return 'Apartment'
    return 'Unknown'

df['property_type'] = df.apply(extract_property_type, axis=1)

def update_property_type(row):
    if row['property_type'] == 'Apartment':
        match = re.search(r'\b[Ff](0?[1-8])\b', row['description'])
        if match:
            return f'F{match.group(1).lstrip("0")}'
    return row['property_type']

df['property_type'] = df.apply(update_property_type, axis=1)

valid_types = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Villa', 'Terrain']

df = df[
    (df['surface'] >= 20) & (df['surface'] <= 1000) &
    (df['price'] >= 1000000) & (df['price'] <= 100000000) &
    (df['town'] != '') &
    (df['property_type'].isin(valid_types))
]

df_encoded = pd.get_dummies(df, columns=['property_type', 'town'], prefix=['type', 'town'])
df_encoded['price'] = np.log1p(df_encoded['price'])
df_encoded['surface'] = np.log1p(df_encoded['surface'])


X = df_encoded[['surface'] + [col for col in df_encoded.columns if col.startswith('type_') or col.startswith('town_')]]
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
mask = (y_train >= Q1 - 1.5 * IQR) & (y_train <= Q3 + 1.5 * IQR)
X_train_filtered = X_train_s[mask]
y_train_filtered = y_train[mask]

mask_test = (y_test >= Q1 - 1.5 * IQR) & (y_test <= Q3 + 1.5 * IQR)
X_test_filtered = X_test_s[mask_test]
y_test_filtered = y_test[mask_test]

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=42)
model.fit(X_train_filtered, y_train_filtered)

#evaluation
y_pred_train = model.predict(X_train_filtered)
y_pred_test_filtered = model.predict(X_test_filtered)

print("\nGradientBoostingRegressor Results:")
print(f"Train R²: {model.score(X_train_filtered, y_train_filtered):.4f}")
print(f"Test R²: {model.score(X_test_filtered, y_test_filtered):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train_filtered, y_pred_train):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test_filtered, y_pred_test_filtered):.4f}")
print(f"Train MSE: {mean_squared_error(y_train_filtered, y_pred_train):.4f}")
print(f"Test MSE: {mean_squared_error(y_test_filtered, y_pred_test_filtered):.4f}")


joblib.dump(model, 'immo_price_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
df_encoded.to_csv('encoded_real_estate_data.csv', index=False)