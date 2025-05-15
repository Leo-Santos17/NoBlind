package com.example.noblind

object Constants {
    // Detector de objetos
    const val MODEL_PATH = "model.tflite"
    const val LABELS_PATH = "labels.txt"

    // Classificadores de roupas
    const val CLOTHING_MODEL_PATH = "clothing_classifier.tflite"
    const val CLOTHING_LABELS_PATH = "clothing_labels.txt"
    const val CALCA_MODEL_PATH = "botton_classifier.tflite"
    const val CALCA_LABELS_PATH = "botton_labels.txt"

    // Adicione mais classificadores aqui conforme necessário
    // const val SHOES_MODEL_PATH = "shoes_classifier.tflite"
    // const val SHOES_LABELS_PATH = "shoes_labels.txt"

    // Tempo de espera antes de classificar (em milissegundos)
    const val CLASSIFICATION_DELAY = 3000L

    // Limiares de confiança para detecção e classificação
    const val DETECTION_CONFIDENCE_THRESHOLD = 0.5f
    const val CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5f
}