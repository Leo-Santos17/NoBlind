package com.example.noblind

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import androidx.core.graphics.scale

class ColorDetector {
    fun detectDominantColor(bitmap: Bitmap, boundingBox: RectF? = null): String {
        // 1. Recortar a região de interesse (se houver bounding box)
        val region = if (boundingBox != null) {
            val x = (boundingBox.left * bitmap.width).toInt().coerceIn(0, bitmap.width)
            val y = (boundingBox.top * bitmap.height).toInt().coerceIn(0, bitmap.height)
            val width = (boundingBox.width() * bitmap.width).toInt().coerceIn(1, bitmap.width - x)
            val height = (boundingBox.height() * bitmap.height).toInt().coerceIn(1, bitmap.height - y)

            Bitmap.createBitmap(bitmap, x, y, width, height)
        } else {
            bitmap
        }

        // 2. Reduzir para uma imagem pequena para análise mais rápida
        val smallBitmap = region.scale(50, 50, false)

        // 3. Extrair cores predominantes usando abordagem simplificada
        val colorCount = mutableMapOf<Int, Int>()
        val pixels = IntArray(smallBitmap.width * smallBitmap.height)
        smallBitmap.getPixels(pixels, 0, smallBitmap.width, 0, 0, smallBitmap.width, smallBitmap.height)

        for (pixel in pixels) {
            // Agrupar cores similares (reduzir precisão para agrupar melhor)
            val simplifiedColor = simplifyColor(pixel)
            colorCount[simplifiedColor] = colorCount.getOrDefault(simplifiedColor, 0) + 1
        }

        // 4. Encontrar a cor mais frequente
        val dominantColor = colorCount.maxByOrNull { it.value }?.key ?: Color.BLACK

        // 5. Converter para nome de cor humano
        return getColorName(dominantColor)
    }

    private fun simplifyColor(color: Int): Int {
        val r = (Color.red(color) / 10) * 10
        val g = (Color.green(color) / 10) * 10
        val b = (Color.blue(color) / 10) * 10
        return Color.rgb(r, g, b)
    }

    private fun getColorName(color: Int): String {
        val hsv = FloatArray(3)
        Color.RGBToHSV(Color.red(color), Color.green(color), Color.blue(color), hsv)

        return when {
            hsv[1] < 0.2 -> when { // Cores pouco saturadas (tons de cinza/branco/preto)
                hsv[2] > 0.9 -> "branco"
                hsv[2] < 0.2 -> "preto"
                else -> "cinza"
            }
            hsv[0] in 0f..15f -> "vermelho"
            hsv[0] in 15f..45f -> "laranja"
            hsv[0] in 45f..70f -> "amarelo"
            hsv[0] in 70f..160f -> "verde"
            hsv[0] in 160f..260f -> "azul"
            hsv[0] in 260f..290f -> "roxo"
            hsv[0] in 290f..330f -> "rosa"
            else -> "vermelho"
        } + when { // Adicionar luminosidade
            hsv[2] < 0.3 -> " escuro"
            hsv[2] > 0.7 -> " claro"
            else -> ""
        }
    }
}