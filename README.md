# API de Predicción de Siniestros de Tránsito

API REST que predice la causa probable de un siniestro de tránsito usando Machine Learning (Random Forest).

## Descripción

API que clasifica la causa de un siniestro en 3 categorías:
- Distracción / Celular
- Exceso Velocidad
- No Respetar Señales

Entrenado con 5,584 registros de Ecuador (2025 Q1).

## Métricas del Modelo

| Métrica | Valor |
|:---|:---:|
| Algoritmo | Random Forest (200 árboles) |
| Accuracy | 92.30% |
| ROC-AUC | 0.9793 |

## Endpoints

- GET / → Información de la API
- GET /health → Estado del servicio
- POST /predecir → Realizar predicción

## Ejemplo de petición

{
  "HORA_NUM": 14,
  "DIA_1": "LUNES",
  "ES_FERIADO": "NO",
  "ZONA": "URBANA",
  "PROVINCIA": "PICHINCHA",
  "TIPO_DE_SINIESTRO": "CHOQUE LATERAL",
  "LESIONADOS": 1,
  "FALLECIDOS": 0,
  "TIENE_MOTOCICLETA": 0,
  "TIENE_AUTOMOVIL": 1
}

## Ejemplo de respuesta

{
  "prediccion": "Distracción / Celular",
  "probabilidades": {
    "Distracción / Celular": 0.84,
    "Exceso Velocidad": 0.125,
    "No Respetar Señales": 0.035
  }
}

## Tecnologías

- Python 3.13
- Flask 3.1
- scikit-learn 1.6.1
- pandas 2.2.3
- numpy 2.1.3
- joblib 1.4.2
- gunicorn 23.0

## Instalación

git clone https://github.com/Balto86/api_siniestros.git
cd api_siniestros
pip install -r requirements.txt
python app.py

La API estará en http://localhost:5000

## Estructura

api_siniestros/
  app.py
  modelo_siniestros.pkl
  escalador.pkl
  columnas_modelo.pkl
  requirements.txt
  Procfile
  datos_prueba.json
  README.md

## Autor

David Fernando Cargua León
Universidad Estatal Amazónica
Minería de Datos - 2026