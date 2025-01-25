import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
from huggingface_hub import login
from sklearn.cluster import KMeans

login(token="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")

class PersonDescriptionSystem:
    def __init__(self):
        # Inicializa os modelos necessários
        self.yolo_model = YOLO('yolo11n.pt')  # Detector de objetos/pessoas
        
        # Modelo para classificação de poses
        self.pose_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.pose_model = AutoModelForImageClassification.from_pretrained("sgdkn/pose-classification-hp")
        
        # Modelo para classificação de roupas
        self.clothing_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.clothing_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

    def detect_person(self, image):
        """Detecta pessoas na imagem usando YOLO"""
        results = self.yolo_model(image)
        # Filtra apenas detecções de pessoas
        person_detections = [det for det in results[0].boxes.data if det[5] == 0]  # classe 0 = pessoa
        return person_detections

    def analyze_pose(self, person_image):
        """Analisa a pose da pessoa (em pé, sentada, etc)"""
        inputs = self.pose_processor(person_image, return_tensors="pt")
        outputs = self.pose_model(**inputs)
        
        # Obtém as probabilidades
        probabilities = outputs.logits.softmax(dim=1)
        predicted_class = probabilities.argmax().item()
        
        # Mapeamento de classes
        pose_mapping = {
            0: "em pé",
            1: "sentado",
            2: "agachado",
            3: "deitado"
        }
        
        return pose_mapping.get(predicted_class, "pose desconhecida")
        """Mapeia valores RGB para nomes de cores"""
        cores = {
            "vermelho": [255, 0, 0],
            "verde": [0, 255, 0],
            "azul": [0, 0, 255],
            "preto": [0, 0, 0],
            "branco": [255, 255, 255],
            "amarelo": [255, 255, 0],
            "roxo": [128, 0, 128],
            "laranja": [255, 165, 0],
            "rosa": [255, 192, 203],
            "marrom": [139, 69, 19]
        }
        
        # Encontra a cor mais próxima
        distancia_minima = float('inf')
        cor_mais_proxima = "desconhecida"
        
        for nome_cor, valor_rgb in cores.items():
            distancia = sum((a - b) ** 2 for a, b in zip(rgb_values, valor_rgb))
            if distancia < distancia_minima:
                distancia_minima = distancia
                cor_mais_proxima = nome_cor
                
        return cor_mais_proxima

    def generate_description(self, image_path):
        """Gera uma descrição completa da pessoa na imagem"""
        # Carrega a imagem
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detecta pessoas
        person_detections = self.detect_person(image)
        
        descriptions = []
        for detection in person_detections:
            # Extrai a região da imagem com a pessoa
            x1, y1, x2, y2 = map(int, detection[:4])
            person_img = image[y1:y2, x1:x2]
            
            # Analisa características
            pose = self.analyze_pose(person_img)
            
            # Gera descrição
            description = f"Pessoa encontrada: {pose}"
            descriptions.append(description)
            
        return descriptions

# Exemplo de uso
if __name__ == "__main__":
    describer = PersonDescriptionSystem()
    # Insira o Caminho da imagem
    descriptions = describer.generate_description("Insira_o_caminho.jpg")
    for desc in descriptions:
        print(desc)