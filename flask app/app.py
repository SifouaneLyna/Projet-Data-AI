from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

model = joblib.load('immo_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
X_train_columns = pd.read_csv('encoded_real_estate_data.csv').drop('price', axis=1).columns

def format_centimes(price):
    centimes = price * 100
    if centimes >= 1_000_000_000:
        return f"{centimes / 1_000_000_000:,.3f} milliard de centimes"
    elif centimes >= 1_000_000:
        return f"{centimes / 1_000_000:,.3f} million de centimes"
    else:
        return f"{centimes:,.0f} centimes"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    errors = {}
    submitted_info = None

    if request.method == 'POST':
        try:
            surface = request.form.get('surface')
            main_property_type = request.form.get('main_property_type')
            property_type = request.form.get('property_type')
            town = request.form.get('town')

            if not surface:
                errors['surface'] = "Surface requise"
            else:
                try:
                    surface = float(surface)
                    if surface < 35:
                        errors['surface'] = "Surface doit être supérieure ou égale à 35 m²"
                    elif surface > 1000:
                        errors['surface'] = "Surface doit être inférieure ou égale à 1000 m²"
                except ValueError:
                    errors['surface'] = "Surface doit être un nombre"

            if not main_property_type:
                errors['main_property_type'] = "Type de propriété requis"

            if main_property_type == 'Apartment' and not property_type:
                errors['property_type'] = "Type d'appartement requis"

            if not town:
                errors['town'] = "Ville requise"


            if not errors:
                input_data = pd.DataFrame({
                    'property_type': [property_type if main_property_type == 'Apartment' else main_property_type],
                    'town': [town],
                    'surface': [surface]
                })

                input_encoded = pd.get_dummies(input_data, columns=['property_type', 'town'], prefix=['type', 'town'])
                input_encoded = input_encoded.reindex(columns=X_train_columns, fill_value=0)
                input_encoded['surface'] = np.log1p(input_encoded['surface'])
                input_scaled = scaler.transform(input_encoded)


                y_pred_log = model.predict(input_scaled)[0]
                predicted_price = np.expm1(y_pred_log)
                min_price = max(0, np.expm1(y_pred_log - 0.2255))
                max_price = np.expm1(y_pred_log + 0.2255)

                # affichage du resultat
                property_display = f"Appartement ({property_type})" if main_property_type == 'Apartment' else main_property_type
                result = {
                    'predicted_price': f"{predicted_price:,.2f}",
                    'min_price': f"{min_price:,.2f}",
                    'max_price': f"{max_price:,.2f}",
                    'predicted_price_centimes': format_centimes(predicted_price),
                    'min_price_centimes': format_centimes(min_price),
                    'max_price_centimes': format_centimes(max_price)
                }
                submitted_info = {
                    'surface': f"{surface:,.1f}",
                    'property_type': property_display,
                    'town': town
                }

                prediction_data = pd.DataFrame({
                    'surface': [surface],
                    'property_type': [property_display],
                    'town': [town],
                    'predicted_price': [predicted_price],
                    'min_price': [min_price],
                    'max_price': [max_price],
                    'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                try:
                    prediction_data.to_csv('predictions.csv', mode='a', header=not pd.io.common.file_exists('predictions.csv'), index=False)
                except Exception as e:
                    errors['general'] = f"Erreur lors de l'enregistrement: {str(e)}"

        except Exception as e:
            errors['general'] = f"Erreur: {str(e)}"

    return render_template('index.html', result=result, errors=errors, submitted_info=submitted_info)

if __name__ == '__main__':
    app.run(debug=True)