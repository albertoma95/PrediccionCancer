import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar datos
def load_data():
    sangre = pd.read_csv("analisis_sangre_dataset.csv", sep=";")
    historial = pd.read_csv("historial_medico.csv")
    cancer = pd.read_csv("analisis_cancer.csv")
    
    df = pd.merge(historial, sangre, left_on='Id', right_on='id', how='inner')
    df = pd.merge(df, cancer, on='id', how='inner')
    
    return df

# Preprocesamiento
def preprocess_data(df):
    df['Sexo'] = df['Sexo'].map({'M': 0, 'F': 1})
    df['Family history'] = df['Family history'].map({'No': 0, 'Yes': 1})
    df['smoke'] = df['smoke'].map({'No': 0, 'Yes': 1})
    df['alcohol'] = df['alcohol'].map({'No': 0, 'Yes': 1})
    df['obesity'] = df['obesity'].map({'Normal': 0, 'Overweight': 1, 'Obese': 2})
    df['diet'] = df['diet'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    df['Screening_History'] = df['Screening_History'].map({'Never': 0, 'Irregular': 1, 'Regular': 2})
    df['Healthcare_Access'] = df['Healthcare_Access'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    df['early_detection'] = df['early_detection'].map({'No': 0, 'Yes': 1})
    df['inflammatory_bowel_disease'] = df['inflammatory_bowel_disease'].map({'No': 0, 'Yes': 1})
    df['relapse'] = df['relapse'].map({'No': 0, 'Yes': 1})
    
    df['cancer_risk'] = (df['cancer_stage'] > 1).astype(int)
    
    return df

# Main
def main():
    df = load_data()
    df_processed = preprocess_data(df)

    X = df_processed.drop(['Id', 'id', 'cancer_stage', 'Survival_Prediction', 'cancer_risk'], axis=1)
    y = df_processed['cancer_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    # Guardar el modelo
    with open("modelo_cancer.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Modelo guardado como 'modelo_cancer.pkl'")

if __name__ == "__main__":
    main()
