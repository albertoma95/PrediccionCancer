import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Cargar el dataset
df = pd.read_csv('../dataset/historial.csv')  # Asegúrate de poner la ruta correcta

# 1. Separar las características (X) y la variable objetivo (y)
X = df.drop('Survival_Prediction', axis=1)  # Usamos todas las columnas excepto la variable objetivo
y = df['Survival_Prediction']  # Variable objetivo

# 2. Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicar SMOTE antes de PCA
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_scaled, y)

print("Distribución después de SMOTE:")
print(y_smote.value_counts())

# 4. Aplicar PCA después de SMOTE para reducir dimensionalidad (manteniendo el 95% de la varianza)
pca = PCA(n_components=0.95)  # Mantener el 95% de la varianza
X_reduced = pca.fit_transform(X_smote)

print(f"Dimensiones originales: {X_scaled.shape}")
print(f"Dimensiones después de PCA: {X_reduced.shape}")

# 5. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_smote, test_size=0.3, random_state=42)

# 6. Definir todos los parámetros posibles para GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  # Parámetro de regularización
    'gamma': ['scale', 'auto', 1, 0.1, 0.01],  # Parámetro gamma para el kernel rbf
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Diferentes tipos de kernel
    'degree': [3, 4, 5],  # Solo para kernel polinomial
    'coef0': [0, 1],  # Solo para kernel polinomial y sigmoid
    'shrinking': [True, False],  # Si usar o no la técnica de "shrinking"
    'tol': [1e-4, 1e-3, 1e-2],  # Tolerancia para el criterio de parada
    'class_weight': [None, 'balanced']  # Balanceo de clases si es necesario
}

# Crear el clasificador SVM
svm = SVC(random_state=42)

# Realizar GridSearchCV para encontrar los mejores parámetros
grid_search = GridSearchCV(svm, param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros
print("Mejores parámetros:", grid_search.best_params_)

# 7. Evaluar el mejor modelo
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 8. Mostrar el reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# 9. Mostrar precisión
print("Precisión:", accuracy_score(y_test, y_pred))

# 10. Validación cruzada para verificar si hay overfitting
cv_scores = cross_val_score(best_model, X_reduced, y_smote, cv=5)  # Validación cruzada con 5 particiones
print(f"Cross-validated accuracy: {cv_scores.mean()}")

# Comparar accuracy en el conjunto de prueba con la validación cruzada
print(f"Accuracy en conjunto de prueba: {accuracy_score(y_test, y_pred)}")
print(f"Desviación estándar de la validación cruzada: {cv_scores.std()}")

