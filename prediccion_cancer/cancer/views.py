from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import joblib
from django.core.files.storage import FileSystemStorage
import json

def index(request):
    return render(request, "index.html")



def predict(request):
    if request.method == 'POST':  # Verificar si es una petición AJAX
            body = json.loads(request.body)

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

            print(data)
            
            # Convertir el diccionario en un DataFrame de pandas
            df = pd.DataFrame(data)

            # Cargar el modelo entrenado (asegúrate de que esté en la ruta correcta)
            model = joblib.load('cancer/models/modelo_svm.pkl')

            # Realizar la predicción usando el modelo
            prediction = model.predict(df)

            # Devolver la predicción en formato JSON para AJAX
            return JsonResponse({'prediction': prediction[0]})
        
    return JsonResponse({'error': 'Petición no válida'}, status=400)

