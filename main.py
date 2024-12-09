import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Carregar o modelo treinado
model = load_model('gesture_recognition_model.h5')

# Inicializar a solução Holistic do MediaPipe
holistic_solution = mp.solutions.holistic
drawing_tools = mp.solutions.drawing_utils

# Definir as classes de gestos
GESTURE_CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2']

# Função para realizar a detecção
def perform_detection(img, model):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False

    detection_result = model.process(img_rgb)

    img_rgb.flags.writeable = True
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return img_bgr, detection_result

# Função para renderizar os pontos-chave da mão
def render_landmarks(img, results):
    if results.right_hand_landmarks:
        drawing_tools.draw_landmarks(
            img, results.right_hand_landmarks, holistic_solution.HAND_CONNECTIONS,
            drawing_tools.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            drawing_tools.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

# Função para extrair os pontos-chave da mão
def extract_hand_landmarks(results):
    if results.right_hand_landmarks:
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten()
    else:
        return np.zeros(21 * 3)

# Inicializar a captura de vídeo
video_capture = cv2.VideoCapture(0)

with holistic_solution.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic_model:
    while True:
        success, frame = video_capture.read()

        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        img_with_detections, detection_results = perform_detection(frame, holistic_model)
        render_landmarks(img_with_detections, detection_results)
        keypoints = extract_hand_landmarks(detection_results).reshape(1, -1, 1)

        if np.any(keypoints):
            prediction = model.predict(keypoints)
            gesture_index = np.argmax(prediction)
            gesture_name = GESTURE_CLASSES[gesture_index]
            confidence = prediction[0][gesture_index]

            cv2.putText(img_with_detections, f'Gesto: {gesture_name} ({confidence:.2f})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Reconhecimento de Gestos', img_with_detections)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
