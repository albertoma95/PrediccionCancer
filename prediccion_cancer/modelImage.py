import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir las rutas de las carpetas con las imágenes
malignant_folder = '../dataset/colon_aca'  # Reemplaza con la ruta de las imágenes malignas
benign_folder = '../dataset/colon_n'   # Reemplaza con la ruta de las imágenes benignas

# 1. Cargar y preprocesar las imágenes
def load_images_from_folder(folder, label, image_size=(224, 224)):  # Usamos 224x224 para reducir el tamaño
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)  # Redimensionar la imagen
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

# Normalizar las imágenes y convertir a float32 para reducir el uso de memoria
images = images / 255.0  # Escalar de 0-255 a 0-1
images = images.astype(np.float32)  # Convertir a float32

# 2. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# 3. Crear el modelo CNN
model = Sequential()

# Capa convolucional
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))  # Usamos 224x224
model.add(MaxPooling2D((2, 2)))

# Capa convolucional
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Aplanar las salidas de las capas convolucionales
model.add(Flatten())

# Capa densa totalmente conectada
model.add(Dense(128, activation='relu'))

# Capa de salida con una neurona (0 o 1 para benigno o maligno)
model.add(Dense(1, activation='sigmoid'))

# 4. Compilar el modelo
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 5. Usar ImageDataGenerator para entrenar con lotes de imágenes
datagen = ImageDataGenerator(rescale=1./255)  # Normalizar las imágenes

# Crear un generador de datos para el conjunto de entrenamiento
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# 6. Entrenar el modelo
history = model.fit(train_generator, epochs=10, validation_data=(X_test, y_test))

# 7. Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 8. Graficar la precisión y la pérdida
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

# 9. Guardar el modelo entrenado
model.save('cancer_model.h5')
print("Modelo guardado como 'cancer_model.h5'")
