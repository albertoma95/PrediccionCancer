import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar los datos
sangre = pd.read_csv(
    r"C:\Users\Salim Tamazirt\Desktop\cancer\data\analisis_sangre_dataset.csv",
    sep=";",
    encoding="latin-1"
)

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sangre.drop('id', axis=1))  # Normalizar los datos

# Clustering para detectar anomalías con KMeans
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters: normal/anormal
sangre['anomaly'] = kmeans.fit_predict(X_scaled)

# Añadir la columna 'is_anomaly' para clasificar las anomalías
sangre['is_anomaly'] = (sangre['anomaly'] == 1).astype(int)  # '1' será considerado anómalo

# Ahora puedes usar la columna 'is_anomaly' como variable objetivo
X = X_scaled
y = sangre['is_anomaly']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

# Entrenamiento del modelo con validación cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
print(f"Precisión promedio con validación cruzada: {cv_scores.mean():.2f}")

# Ajuste de hiperparámetros con GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

# Usar el mejor modelo encontrado con GridSearch
best_model = grid_search.best_estimator_

# Entrenar el modelo con los mejores parámetros
best_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = best_model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}")

# Imprimir el reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Imprimir la matriz de confusión
print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Analizar la importancia de las características
importances = best_model.feature_importances_
print("\nImportancia de las características:")
for i, feature in enumerate(sangre.drop('id', axis=1).columns):
    print(f"{feature}: {importances[i]:.4f}")

# Opcional: Visualización de la matriz de confusión
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anormal"], yticklabels=["Normal", "Anormal"])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
