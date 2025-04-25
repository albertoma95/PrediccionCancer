import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2  # Para regularización L2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping  # Para EarlyStopping
from tensorflow.keras.applications import ResNet50

# Establecer semillas para la reproducibilidad
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Definir las rutas de las carpetas con las imágenes
malignant_folder = '../dataset/colon_aca'  # Reemplaza con la ruta de las imágenes malignas
benign_folder = '../dataset/colon_n'  # Reemplaza con la ruta de las imágenes benignas

# 1. Cargar y preprocesar las imágenes
def load_images_from_folder(folder, label, image_size=(128, 128)):  # Usamos 128x128 para tamaño más grande
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)  # Redimensionar la imagen a 128x128
            images.append(img)
            labels.append(label)  # 0 para benigno, 1 para maligno
    return images, labels

# Cargar las imágenes benignas y malignas
malignant_images, malignant_labels = load_images_from_folder(malignant_folder, 1)
benign_images, benign_labels = load_images_from_folder(benign_folder, 0)

# Combinar las imágenes y etiquetas
images = malignant_images + benign_images
labels = malignant_labels + benign_labels

# Convertir las listas en arrays de NumPy
images = np.array(images)
labels = np.array(labels)

# Normalizar las imágenes
images = images / 255.0  # Escalar de 0-255 a 0-1

# 2. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# 3. Usar ResNet50 preentrenado para transfer learning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Descongelar solo las últimas 5 capas de ResNet50 para el fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-5]:  # Congelar las capas anteriores
    layer.trainable = False

# 4. Construcción del modelo CNN con Regularización L2 y EarlyStopping
model = Sequential()

# Añadir el modelo base de ResNet50
model.add(base_model)

# Aplanar las salidas de las capas convolucionales
model.add(Flatten())

# Aumentar el número de neuronas en la capa densa
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))  # Usamos 256 neuronas
model.add(Dropout(0.5))  # Usar dropout para evitar sobreajuste

# Capa de salida con una neurona (0 o 1 para benigno o maligno)
model.add(Dense(1, activation='sigmoid'))

# 5. Compilar el modelo con RMSprop (sin decay por ahora)
optimizer = RMSprop(learning_rate=0.001)  # Usar un valor fijo de learning rate

# Compilación del modelo con RMSprop
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 6. Configurar EarlyStopping para evitar el sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 7. Entrenar el modelo con más épocas (por ejemplo, 100)
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 8. Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 9. Graficar la precisión y la pérdida
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.legend()
plt.title('Precisión del modelo')
plt.show()

plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.legend()
plt.title('Pérdida del modelo')
plt.show()

# 10. Guardar el modelo entrenado
model.save('cancer_model_with_finetuned_resnet50_and_rmsprop_no_decay.h5')
print("Modelo guardado como 'cancer_model_with_finetuned_resnet50_and_rmsprop_no_decay.h5'")
