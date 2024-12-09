import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout

# Definindo as classes de gestos
GESTURE_CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2']

# Caminho para o dataset
DATA_DIRECTORY = 'dataset'

# Carregando os dados
X, y = [], []

for idx, gesture in enumerate(GESTURE_CLASSES):
    gesture_path = os.path.join(DATA_DIRECTORY, gesture)
    for file in os.listdir(gesture_path):
        file_path = os.path.join(gesture_path, file)
        if file.endswith('.npy'):  # Verificando se o arquivo é um arquivo .npy
            keypoints = np.load(file_path)
            X.append(keypoints)
            y.append(idx)

# Convertendo para arrays numpy
X = np.array(X)
y = np.array(y)

# Convertendo as labels para one-hot encoding
y = to_categorical(y, num_classes=len(GESTURE_CLASSES))

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo os parâmetros do modelo
num_keypoints = X_train.shape[1]
num_classes = len(GESTURE_CLASSES)

# Construindo o modelo LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(num_keypoints, 1)))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumindo o modelo
model.summary()

# Ajustando a forma dos dados
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Treinando o modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Avaliando o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Salvando o modelo
model.save('gesture_recognition_model.h5')
