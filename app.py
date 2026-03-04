from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Charger le modèle Random Forest entraîné avec SMOTE
model = joblib.load("best_model_diabetes.pkl")

# Colonnes utilisées pendant l'entraînement
columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données du formulaire
        features = [float(request.form[col]) for col in columns]
        final_features = pd.DataFrame([features], columns=columns)

        # Prédiction
        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0][1]

        # Interprétation du résultat
        if prediction == 1:
            result = f"⚠ Patient Diabétique (Probabilité: {probability:.2%})"
        else:
            result = f"✅ Patient Non Diabétique (Probabilité: {(1 - probability):.2%})"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        print("Erreur :", e)
        return render_template("index.html", prediction_text="Erreur dans les données")

if __name__ == "__main__":

    #app.run(debug=True)

    #cela rend ton application prête pour un déploiement cloud (Render, Railway, etc.)
    #app.run(host="0.0.0.0", port=5000, debug=True)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    #En production → PAS de debug=True

    #commentaire
    #L’application a été déployée en ligne via la plateforme cloud Render, permettant une accessibilité publique du modèle prédictif à travers une interface web interactive.