package com.example.noblind

object Constants {
    const val MODEL_PATH = "model.tflite"
    const val LABELS_PATH = "labels.txt"

    // Classificador de roupas
    const val CLOTHING_MODEL_PATH = "clothing_classifier.tflite"
    const val CLOTHING_LABELS_PATH = "clothing_labels.txt"

    // Tempo de espera antes de classificar (em milissegundos)
    const val CLASSIFICATION_DELAY = 6000L
}