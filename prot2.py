import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
import math

class ColorIdentifier:
    def __init__(self, n_colors=5, calibration_factor=1.0):
        """
        Inicializa o identificador de cores.
        
        Args:
            n_colors (int): Número de cores a serem extraídas na análise.
            calibration_factor (float): Fator de calibração para ajuste de cores.
        """
        self.n_colors = n_colors
        self.calibration_factor = calibration_factor
        
        # Dicionário com nomes de cores em português
        self.color_names = {
            'black': 'preto',
            'gray': 'cinza',
            'silver': 'prata',
            'white': 'branco',
            'maroon': 'bordô',
            'red': 'vermelho',
            'purple': 'roxo',
            'fuchsia': 'fúcsia',
            'green': 'verde',
            'lime': 'verde-limão',
            'olive': 'verde-oliva',
            'yellow': 'amarelo',
            'navy': 'azul-marinho',
            'blue': 'azul',
            'teal': 'azul-petróleo',
            'aqua': 'água',
            'orange': 'laranja',
            'chocolate': 'marrom',
            'brown': 'marrom',
            'pink': 'rosa',
            'gold': 'dourado',
            'beige': 'bege',
            'ivory': 'marfim',
            'khaki': 'cáqui'
        }
        
        # Paleta de cores estendida para correspondência mais precisa
        self.extended_colors = {
            # Tons de vermelho
            (139, 0, 0): 'vermelho escuro',
            (220, 20, 60): 'vermelho carmesim',
            (255, 0, 0): 'vermelho',
            (255, 99, 71): 'vermelho-coral',
            (255, 127, 80): 'coral',
            
            # Tons de rosa
            (255, 192, 203): 'rosa claro',
            (255, 105, 180): 'rosa choque',
            (219, 112, 147): 'rosa violeta',
            (199, 21, 133): 'rosa médio',
            (255, 20, 147): 'rosa profundo',
            
            # Tons de laranja
            (255, 165, 0): 'laranja',
            (255, 140, 0): 'laranja escuro',
            (255, 69, 0): 'vermelho-alaranjado',
            
            # Tons de amarelo
            (255, 255, 0): 'amarelo',
            (255, 215, 0): 'dourado',
            (238, 232, 170): 'amarelo claro',
            (240, 230, 140): 'caqui',
            
            # Tons de verde
            (0, 128, 0): 'verde',
            (34, 139, 34): 'verde floresta',
            (50, 205, 50): 'verde limão',
            (0, 255, 0): 'verde claro',
            (152, 251, 152): 'verde pálido',
            (0, 250, 154): 'verde-água',
            (60, 179, 113): 'verde-esmeralda',
            (46, 139, 87): 'verde-mar',
            (128, 128, 0): 'verde-oliva',
            (85, 107, 47): 'verde musgo',
            
            # Tons de azul
            (0, 0, 255): 'azul',
            (0, 0, 139): 'azul escuro',
            (0, 0, 128): 'azul-marinho',
            (25, 25, 112): 'azul meia-noite',
            (65, 105, 225): 'azul royal',
            (0, 191, 255): 'azul céu',
            (135, 206, 235): 'azul claro',
            (135, 206, 250): 'azul celeste',
            (70, 130, 180): 'azul aço',
            (100, 149, 237): 'azul cornflower',
            (0, 128, 128): 'azul-petróleo',
            
            # Tons de roxo
            (128, 0, 128): 'roxo',
            (148, 0, 211): 'violeta',
            (153, 50, 204): 'roxo orquídea',
            (186, 85, 211): 'roxo médio',
            (221, 160, 221): 'roxo claro',
            (238, 130, 238): 'violeta claro',
            (218, 112, 214): 'orquídea',
            (216, 191, 216): 'lilás',
            (221, 160, 221): 'rosa-roxo',
            
            # Tons de marrom
            (165, 42, 42): 'marrom',
            (139, 69, 19): 'marrom sela',
            (210, 105, 30): 'chocolate',
            (244, 164, 96): 'marrom claro',
            (160, 82, 45): 'siena',
            (205, 133, 63): 'peru',
            (222, 184, 135): 'bronzeado',
            (210, 180, 140): 'marrom claro',
            
            # Tons de branco
            (255, 255, 255): 'branco',
            (255, 250, 250): 'branco neve',
            (245, 245, 245): 'branco fumaça',
            (240, 255, 240): 'branco-verde',
            (245, 255, 250): 'branco menta',
            (240, 255, 255): 'branco-azul',
            (255, 250, 240): 'branco floral',
            (253, 245, 230): 'branco antigo',
            (255, 245, 238): 'branco-salmão',
            (245, 245, 220): 'bege',
            (255, 228, 196): 'bege antigo',
            (255, 235, 205): 'bege claro',
            (255, 228, 225): 'rosa misty',
            (250, 235, 215): 'marfim antigo',
            (255, 239, 213): 'papaia',
            (255, 218, 185): 'pêssego',
            
            # Tons de cinza
            (0, 0, 0): 'preto',
            (105, 105, 105): 'cinza escuro',
            (128, 128, 128): 'cinza',
            (169, 169, 169): 'cinza médio',
            (192, 192, 192): 'cinza claro',
            (211, 211, 211): 'cinza muito claro',
            (220, 220, 220): 'cinza gainsboro',
            (245, 245, 245): 'cinza branqueado',
            (112, 128, 144): 'cinza ardósia',
            (119, 136, 153): 'cinza ardósia claro',
        }
        
    def calibrate(self, reference_image, reference_color):
        """
        Calibra o sistema com base em uma imagem de referência.
        
        Args:
            reference_image: Imagem de referência com uma cor conhecida.
            reference_color: Valor RGB da cor conhecida na imagem.
            
        Returns:
            float: Fator de calibração atualizado.
        """
        detected_color = self.get_dominant_color(reference_image)
        
        # Calcula a diferença média entre as cores detectadas e as cores de referência
        diff_r = reference_color[0] / max(detected_color[0], 1)
        diff_g = reference_color[1] / max(detected_color[1], 1)
        diff_b = reference_color[2] / max(detected_color[2], 1)
        
        # Atualiza o fator de calibração (média das diferenças)
        self.calibration_factor = (diff_r + diff_g + diff_b) / 3
        
        return self.calibration_factor
    
    def preprocess_image(self, image):
        """
        Pré-processa a imagem para melhorar a detecção de cores.
        
        Args:
            image: Imagem a ser processada (array NumPy).
            
        Returns:
            array: Imagem processada.
        """
        # Converte para RGB se a imagem estiver em BGR (OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Aplica um leve desfoque para reduzir ruído
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Ajusta o contraste e brilho para melhorar a detecção
        alpha = 1.2  # Contraste (1.0-3.0)
        beta = 10    # Brilho (0-100)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image
    
    def get_dominant_color(self, image):
        """
        Extrai a cor dominante de uma imagem.
        
        Args:
            image: Imagem a ser analisada (array NumPy).
            
        Returns:
            tuple: Valores RGB da cor dominante.
        """
        # Pré-processa a imagem
        processed_image = self.preprocess_image(image)
        
        # Reshape da imagem para uma lista de pixels
        pixels = processed_image.reshape(-1, 3)
        
        # Aplicar K-means para encontrar as cores principais
        kmeans = KMeans(n_clusters=self.n_colors, random_state=0)
        kmeans.fit(pixels)
        
        # Obter as cores dos centróides
        colors = kmeans.cluster_centers_
        
        # Contar os pixels em cada cluster
        labels = kmeans.labels_
        count = Counter(labels)
        
        # Encontrar o cluster mais frequente
        dominant_color_idx = count.most_common(1)[0][0]
        dominant_color = colors[dominant_color_idx]
        
        # Aplicar calibração
        calibrated_color = [
            min(255, c * self.calibration_factor) for c in dominant_color
        ]
        
        # Arredondar para valores inteiros
        return tuple(int(c) for c in calibrated_color)
    
    def get_color_name(self, rgb_color):
        """
        Identifica o nome da cor com base nos valores RGB.
        
        Args:
            rgb_color: Tupla com valores RGB (r, g, b).
            
        Returns:
            str: Nome descritivo da cor.
        """
        # Primeiro tentamos encontrar a correspondência na nossa paleta estendida
        min_distance = float('inf')
        closest_color_name = "desconhecido"
        
        for color_rgb, color_name in self.extended_colors.items():
            distance = self._color_distance(rgb_color, color_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_color_name = color_name
        
        # Se a distância for muito grande, vamos tentar encontrar um nome aproximado
        # com base na luminosidade e saturação
        if min_distance > 100:
            r, g, b = rgb_color
            intensity = sum(rgb_color) / 3
            
            # Análise de intensidade para adicionar adjetivos (claro/escuro)
            base_name = closest_color_name.split()[-1]  # Remove possíveis adjetivos anteriores
            
            # Verifica qual canal de cor é dominante
            max_channel = max(rgb_color)
            
            if intensity > 200:
                if "claro" not in closest_color_name:
                    return f"{base_name} claro"
            elif intensity < 80:
                if "escuro" not in closest_color_name:
                    return f"{base_name} escuro"
                    
            # Verifica a saturação
            max_diff = max_channel - min(rgb_color)
            if max_diff < 30 and intensity > 100 and intensity < 200:
                if r > 150 and g > 150 and b > 150:
                    return "cinza claro"
                elif r < 150 and g < 150 and b < 150:
                    return "cinza escuro"
        
        return closest_color_name
    
    def _color_distance(self, color1, color2):
        """
        Calcula a distância euclidiana entre duas cores no espaço RGB.
        
        Args:
            color1: Primeira cor (r, g, b).
            color2: Segunda cor (r, g, b).
            
        Returns:
            float: Distância entre as cores.
        """
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        # Usamos a distância euclidiana ponderada (dando mais peso para certas cores)
        # baseado na sensibilidade do olho humano
        return math.sqrt(2*(r1-r2)**2 + 4*(g1-g2)**2 + 3*(b1-b2)**2)
    
    def analyze_clothing_color(self, clothing_image):
        """
        Analisa a cor predominante da peça de roupa.
        
        Args:
            clothing_image: Imagem da peça de roupa (array NumPy).
            
        Returns:
            dict: Dicionário contendo a cor dominante em RGB e o nome descritivo.
        """
        # Obtém a cor dominante
        dominant_color = self.get_dominant_color(clothing_image)
        
        # Obtém o nome da cor
        color_name = self.get_color_name(dominant_color)
        
        return {
            'rgb': dominant_color,
            'name': color_name
        }
    
    def handle_varied_lighting(self, clothing_image):
        """
        Processa a imagem para lidar com variações de iluminação.
        
        Args:
            clothing_image: Imagem da peça de roupa (array NumPy).
            
        Returns:
            dict: Dicionário contendo a cor dominante em RGB e o nome descritivo.
        """
        # Converte para o espaço de cor HSV que é menos sensível à iluminação
        if len(clothing_image.shape) == 3 and clothing_image.shape[2] == 3:
            hsv_image = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2HSV)
        else:
            return self.analyze_clothing_color(clothing_image)
        
        # Normaliza a iluminação (canal V do HSV)
        h, s, v = cv2.split(hsv_image)
        
        # Aplica equalização de histograma apenas no canal V
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        
        # Mescla novamente os canais
        hsv_equalized = cv2.merge([h, s, v_eq])
        
        # Converte de volta para RGB
        rgb_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)
        
        # Analisa a cor usando a imagem equalizada
        return self.analyze_clothing_color(rgb_equalized)

# Função para uso com o YOLO
def identify_clothing_color(cropped_image, calibration_image=None, reference_color=None):
    """
    Identifica a cor predominante em uma peça de roupa detectada.
    
    Args:
        cropped_image: Imagem recortada da peça de roupa (região delimitada pelo YOLO).
        calibration_image (opcional): Imagem de referência para calibração.
        reference_color (opcional): Cor de referência conhecida para calibração.
        
    Returns:
        dict: Informações sobre a cor predominante.
    """
    # Inicializa o identificador de cores
    color_identifier = ColorIdentifier(n_colors=5)
    
    # Calibra o sistema se fornecidos os parâmetros de calibração
    if calibration_image is not None and reference_color is not None:
        color_identifier.calibrate(calibration_image, reference_color)
    
    # Analisa a cor considerando variações de iluminação
    color_info = color_identifier.handle_varied_lighting(cropped_image)
    
    return color_info


# Exemplo de uso com YOLO:

# Importar bibliotecas YOLO
import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('yolov8n.pt')

# Carregar uma imagem
image = cv2.imread("images/camisa2.webp")

# Executar a detecção
results = model(image)

# Processar cada detecção
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    
    for i, box in enumerate(boxes):
        # Verificar se a classe detectada é uma peça de roupa (conforme o modelo usado)
        class_id = int(classes[i])
        if class_id in [0, 1, 2, 3]:  # IDs de classes correspondentes a peças de roupa
            # Extrair as coordenadas da caixa delimitadora
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Recortar a região da peça de roupa
            clothing_crop = image[y1:y2, x1:x2]
            
            # Identificar a cor predominante
            color_info = identify_clothing_color(clothing_crop)
            
            # Exibir informações
            print(f"Peça de roupa detectada com cor predominante: {color_info['name']} - RGB: {color_info['rgb']}")
            
            # Desenhar caixa delimitadora na imagem original
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adicionar texto com o nome da cor
            cv2.putText(image, color_info['name'], (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostrar a imagem com as detecções e cores
cv2.imshow('Detecção de Peças de Roupa e Cores', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
