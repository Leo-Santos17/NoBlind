import cv2
from ultralytics import YOLO

# Modelos
yolo_model = YOLO('yolo11n.pt')

# Caminho da imagem
caminho_imagem = "images/test_pose2.jpg"



"""Não alterar"""
# Detectar pessoa
def det_person(image):
    results = yolo_model(image)
    person_detections = [det for det in results[0].boxes.data if det[5] == 0]
    person_detections = extract_image(person_detections)
    return person_detections
# Extrai a região da imagem com a pessoa
def extract_image(p):
    for detection in p:
        x1, y1, x2, y2 = map(int, detection[:4])
        person_img = image[y1:y2, x1:x2]
    return person_img
image = cv2.imread(caminho_imagem)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
person_detections = det_person(caminho_imagem)
print(person_detections)