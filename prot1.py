import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Modelos
yolo_model = YOLO('yolo11n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# Caminho da imagem
caminho_imagem = "images/test_pose2.jpg"



"""Não alterar"""
# Detectar pessoa
def det_person(image):
    results = yolo_model(image)
    person_detections = [det for det in results[0].boxes.data if det[5] == 0]
    person_detections = extract_image(person_detections)
    posture = analyse_pose(person_detections)
    return posture
# Extrai a região da imagem com a pessoa
def extract_image(p):
    for detection in p:
        x1, y1, x2, y2 = map(int, detection[:4])
        person_img = image[y1:y2, x1:x2]
    return person_img
# Pose
def analyse_pose(image):
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Pontos chaves relevantes
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP] # Femur
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE] # Joelho
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE] # Tornozelo

        angle = angle_calc(hip,knee,ankle)

        if angle >150:
            return "Em pé"
        else:
            return "Sentada"
        
        return "Não foi possível determinar"

def angle_calc(p1,p2,p3):
    p1 = np.array([p1.x, p1.y])
    p2 = np.array([p2.x, p2.y])
    p3 = np.array([p3.x, p3.y])

    v1 = p1-p2
    v2 = p3-p2

    cos_angle = np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle

    
image = cv2.imread(caminho_imagem)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pessoa = det_person(caminho_imagem)
print(pessoa)