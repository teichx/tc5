import cv2
import numpy as np
import os
import mediapipe as mp

# Importa a solução Holistic do MediaPipe para detecção de landmarks do corpo
holistic_solution = mp.solutions.holistic
# Importa as ferramentas de desenho do MediaPipe para desenhar os landmarks
drawing_tools = mp.solutions.drawing_utils

# Variável que será usada para definir os gestos que serão capturados
Gesto = "?"

def perform_detection(img, model):
    # Converte a imagem de BGR para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img_rgb.flags.writeable = False  # Torna a imagem não editável para melhorar a performance

    # Processa a imagem com o modelo Holistic
    detection_result = model.process(img_rgb)  

    img_rgb.flags.writeable = True  # Torna a imagem editável novamente
    # Converte a imagem de volta de RGB para BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  

    return img_bgr, detection_result

def render_landmarks(img, results):
    # Desenha os landmarks da mão direita se forem detectados
    if results.right_hand_landmarks:
        drawing_tools.draw_landmarks(img, results.right_hand_landmarks, holistic_solution.HAND_CONNECTIONS,
                                     drawing_tools.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                     drawing_tools.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_hand_landmarks(results):
    # Extrai as coordenadas dos landmarks da mão direita se forem detectados
    if results.right_hand_landmarks:
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten()
    else:
        return np.zeros(21 * 3)

# Diretório onde os dados serão salvos
DATA_DIRECTORY = os.path.join('dataset')

# Número de imagens por classe de gesto
IMAGES_PER_CLASS = 200

# Cria diretórios para cada gesto
for gesture in Gesto:
    os.makedirs(os.path.join(DATA_DIRECTORY, gesture), exist_ok=True)

# Captura de vídeo da webcam
video_capture = cv2.VideoCapture(0)

# Inicia o modelo Holistic com confiança mínima de detecção e rastreamento
with holistic_solution.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic_model:
    for gesture in Gesto:
        for image_index in range(IMAGES_PER_CLASS):
            success, frame = video_capture.read()

            # Redimensiona o frame para 640x480 pixels
            frame = cv2.resize(frame, (640, 480))
            # Realiza a detecção de landmarks na imagem
            img_with_detections, detection_results = perform_detection(frame, holistic_model)
            # Desenha os landmarks na imagem
            render_landmarks(img_with_detections, detection_results)

            # Exibe a imagem com os landmarks
            cv2.imshow('', img_with_detections)
            # Extrai os landmarks da mão direita
            keypoints = extract_hand_landmarks(detection_results)
            # Caminho para salvar os dados dos landmarks
            npy_save_path = os.path.join(DATA_DIRECTORY, gesture, str(image_index))
            # Salva os landmarks em um arquivo .npy
            np.save(npy_save_path, keypoints)

            # Sai do loop se a tecla 'q' for pressionada
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

# Libera a captura de vídeo e fecha todas as janelas do OpenCV
video_capture.release()
cv2.destroyAllWindows()
