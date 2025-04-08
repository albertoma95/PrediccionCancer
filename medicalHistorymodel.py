import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Cargar datos
historial = pd.read_csv("historial_medico.csv")

# Preprocesamiento
le = LabelEncoder()
categorical_cols = ['Sexo', 'Family history', 'smoke', 'alcohol', 'obesity', 'diet', 
                   'Screening_History', 'Healthcare_Access']
for col in categorical_cols:
    historial[col] = le.fit_transform(historial[col])

# Variable objetivo
historial['Survival_Prediction'] = le.fit_transform(historial['Survival_Prediction'])

# Dividir datos
X = historial.drop(['Id', 'Survival_Prediction'], axis=1)
y = historial['Survival_Prediction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model_historial = RandomForestClassifier(n_estimators=100, random_state=42)
model_historial.fit(X_train, y_train)

# Evaluación
y_pred = model_historial.predict(X_test)
print(f"Accuracy (Historial Médico - Supervivencia): {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))