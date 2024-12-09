import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Definindo as classes de gestos
GESTURE_CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2']

# Caminho para o dataset
DATA_DIRECTORY = 'dataset'

# Funções de data augmentation
def jitter_keypoints(keypoints, sigma=0.01):
    noise = np.random.normal(0, sigma, keypoints.shape)
    return keypoints + noise

def scale_keypoints(keypoints, scale=0.1):
    factor = 1 + np.random.uniform(-scale, scale)
    return keypoints * factor

def translate_keypoints(keypoints, translate=0.1):
    if len(keypoints.shape) == 1:
        num_keypoints = keypoints.shape[0] // 3
        keypoints = keypoints.reshape((num_keypoints, 3))
    shift = np.random.uniform(-translate, translate, keypoints.shape)
    translated_keypoints = keypoints + shift
    return translated_keypoints.flatten()

def rotate_keypoints(keypoints, angle_range=(-15, 15)):
    if len(keypoints.shape) == 1:
        num_keypoints = keypoints.shape[0] // 3
        keypoints = keypoints.reshape((num_keypoints, 3))
    angle = np.random.uniform(angle_range[0], angle_range[1])
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    rotated_keypoints = np.dot(keypoints[:, :2], rotation_matrix.T)
    keypoints[:, :2] = rotated_keypoints
    return keypoints.flatten()

def augment_keypoints(keypoints):
    augmented_keypoints = jitter_keypoints(keypoints)
    augmented_keypoints = scale_keypoints(augmented_keypoints)
    augmented_keypoints = translate_keypoints(augmented_keypoints)
    augmented_keypoints = rotate_keypoints(augmented_keypoints)
    return augmented_keypoints

# Carregando e augmentando os dados
X, y = [], []

for idx, gesture in enumerate(GESTURE_CLASSES):
    gesture_path = os.path.join(DATA_DIRECTORY, gesture)
    for file in os.listdir(gesture_path):
        if file.endswith('.npy'):  # Verificando se o arquivo é um arquivo .npy
            file_path = os.path.join(gesture_path, file)
            keypoints = np.load(file_path, allow_pickle=True)  # Carregando com allow_pickle=True
            
            # Adicionando os dados originais
            X.append(keypoints)
            y.append(idx)
            
            # Aplicando data augmentation e adicionando dados augmentados
            for _ in range(5):  # Multiplicando por 5 o número de exemplos augmentados
                augmented_keypoints = augment_keypoints(keypoints)
                X.append(augmented_keypoints)
                y.append(idx)

# Convertendo para arrays numpy
X = np.array(X)
y = np.array(y)

# Verificando se o dataset foi aumentado corretamente
print(f'Tamanho do dataset original: {len(GESTURE_CLASSES) * 600}')
print(f'Tamanho do dataset após augmentation: {len(X)}')

# Convertendo as labels para one-hot encoding
y = to_categorical(y, num_classes=len(GESTURE_CLASSES))

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)