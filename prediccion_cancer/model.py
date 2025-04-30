import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Para guardar el modelo
import numpy as np

# Cargar el dataset
df = pd.read_csv('../dataset/final.csv')

# 1. Separar las características (X) y la variable objetivo (y)
X = df.drop(['Survival_Prediction', 'id'], axis=1)  # Eliminar 'Survival_Prediction' y 'id'
y = df['Survival_Prediction']  # Variable objetivo

# 2. Convertir 'Yes'/'No' a 1/0 en la variable objetivo
le = LabelEncoder()
y = le.fit_transform(y)  # Transforma 'Yes' -> 1, 'No' -> 0

# Variables categóricas
categorical_columns = ['Sexo', 'Family history', 'smoke', 'alcohol', 'obesity', 'diet', 
                       'Screening_History', 'Healthcare_Access', 'cancer_stage', 'early_detection', 
                       'inflammatory_bowel_disease', 'relapse']

# Preprocesamiento de datos: OneHotEncoding para las variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un pipeline que combine preprocesamiento y entrenamiento del modelo
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),  # Escalar los datos
    ('model', xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        use_label_encoder=False, 
        random_state=42,
        max_depth=3, 
        learning_rate=0.005, 
        n_estimators=100, 
        subsample=0.7, 
        colsample_bytree=0.7, 
        gamma=0.5, 
        lambda_=0, 
        alpha=1,
        scale_pos_weight=ratio 
    ))
])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Entrenar el modelo
model_pipeline.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model_pipeline.predict(X_test)

# Mostrar el reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Mostrar precisión
print("Precisión:", accuracy_score(y_test, y_pred))

# Guardar el modelo entrenado en un archivo .pkl
joblib.dump(model_pipeline, 'xgb_model.pkl')
print("Modelo guardado como 'xgb_model.pkl'")
