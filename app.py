# =========================================================
# API DE PREDICCIÓN DE SINIESTROS DE TRÁNSITO
# =========================================================
# Autor: David Fernando Cargua León
# Descripción: API REST que predice la causa probable de un
#              siniestro de tránsito usando un modelo Random Forest.
# =========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback

# =========================================================
# 1. CREAR LA APP FLASK
# =========================================================
app = Flask(__name__)

# Habilitar CORS para permitir peticiones desde el navegador
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# =========================================================
# 2. CARGAR EL MODELO, ESCALADOR Y COLUMNAS
# =========================================================
print("🔄 Cargando modelo...")
modelo = joblib.load('modelo_siniestros.pkl')
escalador = joblib.load('escalador.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')
print("✅ Modelo cargado correctamente")
print(f"   Clases: {modelo.classes_}")
print(f"   Total de columnas: {len(columnas_modelo)}")


# =========================================================
# 3. ENDPOINT RAÍZ - Información de la API
# =========================================================
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


# =========================================================
# 4. ENDPOINT HEALTH - Verificar estado
# =========================================================
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "modelo_cargado": True,
        "clases": list(modelo.classes_)
    })


# =========================================================
# 5. ENDPOINT PREDECIR - Realizar predicción
# =========================================================
@app.route('/predecir', methods=['POST', 'OPTIONS'])
def predecir():
    # Manejar preflight de CORS
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    
    datos = None
    df_encoded = None
    
    try:
        # Obtener datos del request
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

        # Reordenar las columnas en el mismo orden que el modelo
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
        print("❌ ERROR:", error_detallado)
        return jsonify({
            "error": str(e),
            "tipo": type(e).__name__,
            "traceback": error_detallado
        }), 500


# =========================================================
# 6. EJECUTAR LA APP
# =========================================================
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)