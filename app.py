# =========================================================
# API DE PREDICCIÓN DE SINIESTROS DE TRÁNSITO
# =========================================================
# Autor: David Fernando Cargua León
# Descripción: API REST que predice la causa probable de un
#              siniestro de tránsito.
# =========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback

# ✅ CREAR LA APP PRIMERO (esto faltaba)
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Cargar el modelo, escalador y columnas
print("🔄 Cargando modelo...")
modelo = joblib.load('modelo_siniestros.pkl')
escalador = joblib.load('escalador.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')
print("✅ Modelo cargado correctamente")
print(f"   Clases del modelo: {modelo.classes_}")
print(f"   Total de columnas: {len(columnas_modelo)}")


@app.route('/')
def home():
    return jsonify({
        "mensaje": "API de Predicción de Siniestros de Tránsito",
        "version": "1.0",
        "autor": "David Fernando Cargua León",
        "endpoints": {
            "/": "GET - Información de la API",
            "/health": "GET - Verifica el estado",
            "/predecir": "POST - Realiza una predicción"
        }
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok", "modelo_cargado": True})


@app.route('/predecir', methods=['POST', 'OPTIONS'])
def predecir():
    # Manejar preflight de CORS
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    
    datos = None
    df_encoded = None
    
    try:
        datos = request.get_json()
        
        if not datos:
            return jsonify({"error": "No se recibieron datos"}), 400

        # Crear DataFrame con los datos de entrada
        df_input = pd.DataFrame([datos])

        # Aplicar One-Hot Encoding
        df_encoded = pd.get_dummies(df_input, drop_first=True)

        # Asegurar que tenga TODAS las columnas del modelo
        for col in columnas_modelo:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reordenar las columnas
        df_encoded = df_encoded[columnas_modelo]

        # Escalar los datos
        X_escalado = escalador.transform(df_encoded)

        # Realizar la predicción
        prediccion = modelo.predict(X_escalado)[0]
        probabilidades = modelo.predict_proba(X_escalado)[0]

        # Construir la respuesta
        respuesta = {
            "prediccion": str(prediccion),
            "probabilidades": {
                str(clase): round(float(prob), 4)
                for clase, prob in zip(modelo.classes_, probabilidades)
            }
        }

        return jsonify(respuesta)

    except Exception as e:
        error_detallado = traceback.format_exc()
        print("❌ ERROR EN /predecir:")
        print(error_detallado)
        return jsonify({
            "error": str(e),
            "tipo": type(e).__name__,
            "traceback": error_detallado,
            "columnas_esperadas_primeras_10": list(columnas_modelo[:10]) if columnas_modelo else None,
            "columnas_generadas": list(df_encoded.columns)[:20] if df_encoded is not None else None
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)