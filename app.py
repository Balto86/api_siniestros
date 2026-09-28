@app.route('/predecir', methods=['POST', 'OPTIONS'])
def predecir():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    
    datos = None
    df_encoded = None
    
    try:
        datos = request.get_json()
        
        if not datos:
            return jsonify({"error": "No se recibieron datos"}), 400

        # Crear DataFrame
        df_input = pd.DataFrame([datos])
        
        # One-Hot Encoding
        df_encoded = pd.get_dummies(df_input, drop_first=True)

        # Agregar columnas faltantes
        for col in columnas_modelo:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reordenar columnas
        df_encoded = df_encoded[columnas_modelo]

        # Escalar
        X_escalado = escalador.transform(df_encoded)

        # Predecir
        prediccion = modelo.predict(X_escalado)[0]
        probabilidades = modelo.predict_proba(X_escalado)[0]

        respuesta = {
            "prediccion": str(prediccion),
            "probabilidades": {
                str(clase): round(float(prob), 4)
                for clase, prob in zip(modelo.classes_, probabilidades)
            }
        }

        return jsonify(respuesta)

    except Exception as e:
        import traceback
        error_detallado = traceback.format_exc()
        
        return jsonify({
            "error": str(e),
            "tipo": type(e).__name__,
            "traceback": error_detallado,
            "columnas_esperadas_primeras_10": list(columnas_modelo[:10]),
            "columnas_generadas": list(df_encoded.columns) if df_encoded is not None else None
        }), 500