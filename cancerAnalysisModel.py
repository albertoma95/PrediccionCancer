import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Cargar datos
cancer = pd.read_csv("analisis_cancer.csv")

# Preprocesamiento
cancer['early_detection'] = cancer['early_detection'].map({'No': 0, 'Yes': 1})
cancer['inflammatory_bowel_disease'] = cancer['inflammatory_bowel_disease'].map({'No': 0, 'Yes': 1})
cancer['relapse'] = cancer['relapse'].map({'No': 0, 'Yes': 1})

# Dividir datos
X = cancer.drop(['id', 'relapse'], axis=1)
y = cancer['relapse']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo (usamos Gradient Boosting para datos desbalanceados)
model_cancer = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
model_cancer.fit(X_train, y_train)

# Evaluación
y_pred = model_cancer.predict(X_test)
y_proba = model_cancer.predict_proba(X_test)[:, 1]
print(f"Accuracy (Cáncer - Recaída): {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")