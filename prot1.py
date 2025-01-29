import cv2
from ultralytics import YOLO

yolo_model = YOLO('yolo11n.pt')

# Caminho da imagem
caminho_imagem = "images/test_pose2.jpg"


def person(image):
    results = yolo_model(image)
    person_detections = [det for det in results[0].boxes.data if det[5] == 0]
    return person_detections


image = cv2.imread(caminho_imagem)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



person_detections = person(caminho_imagem)
print(person_detections)
descricao = []
for detection in person_detections:
    # Extrai a região da imagem com a pessoa
    x1, y1, x2, y2 = map(int, detection[:4])
    person_img = image[y1:y2, x1:x2]
    print(descricao)
