from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import joblib
from django.core.files.storage import default_storage
import json
from tensorflow.keras.models import load_model
import cv2
import base64
import numpy as np

def index(request):
    return render(request, "index.html")



def predict(request):
    if request.method == 'POST':  # Verificar si es una petición AJAX
        body = json.loads(request.body)

        # Obtener la imagen en base64 desde el JSON
        imagen_base64 = body.get('imagen_base64')
        if not imagen_base64:
            return JsonResponse({'error': 'No se ha recibido ninguna imagen'}, status=400)

        # Decodificar la imagen en base64
        img_data = base64.b64decode(imagen_base64.split(',')[1])  # Eliminar el prefijo 'data:image/jpeg;base64,' si existe
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))  # Redimensionar la imagen a 128x128
        img = img / 255.0  # Normalizar la imagen
        img = img.reshape(1, 128, 128, 3)  # Ajustar la forma de la imagen para el modelo

        # Cargar el modelo de imágenes entrenado
        model_image = load_model('cancer/models/cnn_model.h5')

        # Realizar la predicción usando el modelo de imágenes
        prediction_image = model_image.predict(img)
        prediction_image = int(prediction_image[0] > 0.5)  # Convertir a 0 o 1

        # Preparar los datos para el modelo tabular
        data = {
            'Sexo': [body.get('Sexo')],
            'Age': [float(body.get('Age'))],
            'Family history': [body.get('Family history')],
            'smoke': [body.get('smoke')],
            'alcohol': [body.get('alcohol')],
            'obesity': [body.get('obesity')],
            'diet': [body.get('diet')],
            'Screening_History': [body.get('Screening_History')],
            'Healthcare_Access': [body.get('Healthcare_Access')],
            'cancer_stage': [body.get('cancer_stage')],
            'tumor_size': [float(body.get('tumor_size'))],
            'early_detection': [body.get('early_detection')],
            'inflammatory_bowel_disease': [body.get('inflammatory_bowel_disease')],
            'relapse': [body.get('relapse')],
        }

        # Convertir el diccionario en un DataFrame de pandas
        df = pd.DataFrame(data)

        # Cargar el modelo tabular entrenado
        model_tabular = joblib.load('cancer/models/xgb_model.pkl')

        # Realizar la predicción usando el modelo tabular
        prediction_tabular = model_tabular.predict(df)
        prediction_tabular = int(prediction_tabular[0])  # Asegúrate de obtener el valor entero

        # Hacer una media ponderada entre las predicciones
        weight_tabular = 0.4
        weight_image = 0.6

        final_prediction = (prediction_tabular * weight_tabular) + (prediction_image * weight_image)
        final_prediction = 1 if final_prediction >= 0.5 else 0  # Si la media ponderada es mayor o igual a 0.5, consideramos cáncer maligno

        # Devolver la predicción en formato JSON para AJAX
        return JsonResponse({'prediction': final_prediction})
        
    return JsonResponse({'error': 'Petición no válida'}, status=400)

